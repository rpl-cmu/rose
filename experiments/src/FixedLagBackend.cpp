#include "FixedLagBackend.h"

// ------------------------- Private Methods ------------------------- //
void FixedLagBackend::addIMUEstimate(gtsam::CombinedImuFactor::shared_ptr factor) {
    uint64_t prevStateIdx = gtsam::Symbol(factor->key1()).index();
    uint64_t newStateIdx = gtsam::Symbol(factor->key3()).index();

    gtsam::Pose3 prevX = state_.at<gtsam::Pose3>(X(prevStateIdx));
    gtsam::Velocity3 prevV = state_.at<gtsam::Velocity3>(V(prevStateIdx));
    gtsam::imuBias::ConstantBias prevB = state_.at<gtsam::imuBias::ConstantBias>(B(prevStateIdx));

    gtsam::NavState newStateEstimate =
        factor->preintegratedMeasurements().predict(gtsam::NavState(prevX, prevV), prevB);

    state_.insert(X(newStateIdx), newStateEstimate.pose());
    state_.insert(V(newStateIdx), newStateEstimate.velocity());
    state_.insert(B(newStateIdx), prevB);
}

void FixedLagBackend::addWheel2Estimate(rose::WheelFactor2::shared_ptr factor) {
    uint64_t prevStateIdx = gtsam::Symbol(factor->key1()).index();
    uint64_t newStateIdx = gtsam::Symbol(factor->key2()).index();

    if (!state_.exists(X(newStateIdx))) {
        gtsam::Pose3 prevX = state_.at<gtsam::Pose3>(X(prevStateIdx));
        gtsam::Pose3 newStateEstimate = factor->predict(prevX);
        state_.insert(X(newStateIdx), newStateEstimate);
    }
}

void FixedLagBackend::addWheel3Estimate(rose::WheelFactor3::shared_ptr factor) {
    uint64_t prevStateIdx = gtsam::Symbol(factor->key1()).index();
    uint64_t newStateIdx = gtsam::Symbol(factor->key2()).index();
    gtsam::Key slipKey = factor->key3();

    gtsam::Vector2 slip = gtsam::Vector2::Zero();
    if (!state_.exists(slipKey)) {
        state_.insert(slipKey, slip);
    }

    if (!state_.exists(X(newStateIdx))) {
        gtsam::Pose3 prevX = state_.at<gtsam::Pose3>(X(prevStateIdx));
        gtsam::Pose3 newStateEstimate = factor->predict(prevX, slip);
        state_.insert(X(newStateIdx), newStateEstimate);
    }
}

void FixedLagBackend::addWheel4IntrEstimate(rose::WheelFactor4Intrinsics::shared_ptr factor) {
    uint64_t prevStateIdx = gtsam::Symbol(factor->key1()).index();
    uint64_t newStateIdx = gtsam::Symbol(factor->key3()).index();

    gtsam::Pose3 prevX = state_.at<gtsam::Pose3>(X(prevStateIdx));
    gtsam::Vector3 prevI = state_.at<gtsam::Vector3>(I(prevStateIdx));
    // std::cout << prevI.transpose() << std::endl;

    gtsam::Pose3 newStateEstimate = factor->predict(prevX, prevI);

    if (!state_.exists(X(newStateIdx))) {
        state_.insert(X(newStateIdx), newStateEstimate);
    }
    if (!state_.exists(I(newStateIdx))) {
        state_.insert(I(newStateIdx), prevI);
    }
}

void FixedLagBackend::addWheel5Estimate(rose::WheelFactor5::shared_ptr factor) {
    uint64_t prevStateIdx = gtsam::Symbol(factor->key1()).index();
    uint64_t newStateIdx = gtsam::Symbol(factor->key3()).index();
    gtsam::Key slipKey = factor->key5();

    gtsam::Pose3 prevX = state_.at<gtsam::Pose3>(X(prevStateIdx));
    gtsam::Vector3 prevI = state_.at<gtsam::Vector3>(I(prevStateIdx));

    gtsam::Vector2 slip = gtsam::Vector2::Zero();
    if (!state_.exists(slipKey)) {
        state_.insert(slipKey, slip);
    }

    if (!state_.exists(X(newStateIdx))) {
        gtsam::Pose3 newStateEstimate = factor->predict(prevX, prevI, slip);
        state_.insert(X(newStateIdx), newStateEstimate);
    }
    if (!state_.exists(I(newStateIdx))) {
        state_.insert(I(newStateIdx), prevI);
    }
}

void FixedLagBackend::addLandmarkEstimate(StereoFactor::shared_ptr factor) {
    gtsam::Key poseKey = factor->key1();
    gtsam::Key landmarkKey = factor->key2();

    if (!state_.exists(poseKey)) {
        uint64_t poseIdx = gtsam::Symbol(poseKey).index();
        state_.insert(poseKey, state_.at<gtsam::Pose3>(X(poseIdx - 1)));
    }

    if (!state_.exists(landmarkKey)) {
        gtsam::Pose3 body_P_sensor = factor->body_P_sensor().get_value_or(gtsam::Pose3::Identity());
        gtsam::StereoCamera camera(state_.at<gtsam::Pose3>(poseKey) * body_P_sensor, factor->calibration());
        gtsam::Point3 estimate = camera.backproject(factor->measured());
        state_.insert(landmarkKey, estimate);
    }
}

template <class PriorFactorPointer> void FixedLagBackend::addPriorEstimate(PriorFactorPointer factor) {
    if (!state_.exists(factor->key())) {
        state_.insert(factor->key(), factor->prior());
    }
}

bool FixedLagBackend::getStateToMarg(gtsam::Key &stateToMarg, bool &isKeyframe) {
    bool needToMarg = false;

    // Check which frame to marginalize, if any
    if (regFrames_.size() > regframeNum_) {
        bool makeRegFrameKeyFrame;
        uint64_t transState = regFrames_.front();
        regFrames_.pop();

        // Determine if last regframe should be a keyframe
        if (keyFrames_.size() == 0) {
            makeRegFrameKeyFrame = true;
        } else {
            uint64_t frontKey = keyFrames_.back();
            float distTraveled = (state_.at<gtsam::Pose3>(X(frontKey)).translation() -
                                  state_.at<gtsam::Pose3>(X(transState)).translation())
                                     .norm();
            makeRegFrameKeyFrame = (distTraveled >= keyframeDist_);
        }

        // If this regular frame shouldn't be a keyframe, we'll marginalize it
        if (!makeRegFrameKeyFrame) {
            stateToMarg = X(transState);
            needToMarg = true;
            isKeyframe = false;
        }
        // Otherwise, if we have full keyframes, marg last keyframe
        else {
            keyFrames_.push(transState);
            if (keyFrames_.size() > keyframeNum_) {
                stateToMarg = X(keyFrames_.front());
                keyFrames_.pop();
                needToMarg = true;
                isKeyframe = true;
            }
        }
    }

    return needToMarg;
}

bool FixedLagBackend::isDegenerate(gtsam::NonlinearFactor::shared_ptr factor) {
    auto stereo = boost::dynamic_pointer_cast<StereoFactor>(factor);
    gtsam::Matrix info = stereo->linearize(state_)->information();
    return (info.sum() < 1e-1);
}

void FixedLagBackend::getLMMargDrop(gtsam::Key stateToMarg, const gtsam::VariableIndex &variableIndex,
                                    gtsam::KeySet &keysToMarg, gtsam::KeySet &keysToDrop, FactorSet &facsToMarg,
                                    FactorSet &facsToDrop) {
    gtsam::KeySet lmConnected = mapKey2Key_[stateToMarg];

    for (gtsam::Key lm : lmConnected) {
        // If a landmark is only connected to this state, variable needs to be removed in some way
        const auto &slots = variableIndex[lm];
        if (mapKey2Key_[lm].size() == 1) {
            // If it's also not connected to the marginal prior && is degenerate -> drop it
            if (slots.size() == 1 && isDegenerate(graph_.at(slots[0]))) {
                keysToDrop.insert(lm);
                facsToDrop.insert(slots[0]);
                // If not degenerate, marginalize it -> add it's factors to markov blanket
            } else {
                keysToMarg.insert(lm);
                facsToMarg.insert(slots.begin(), slots.end());
            }
            mapKey2Key_.erase(lm);
            // If it's connected to multiple states, leave the variable be
        } else {
            mapKey2Key_[lm].erase(stateToMarg);
        }
    }

    mapKey2Key_.erase(stateToMarg);
}

void FixedLagBackend::getOdomMargDrop(gtsam::Key stateToMarg, const gtsam::VariableIndex &variableIndex,
                                      gtsam::KeySet &keysToMarg, gtsam::KeySet &keysToDrop, FactorSet &facsToMarg,
                                      FactorSet &facsToDrop) {
    uint64_t stateIdx = gtsam::Symbol(stateToMarg).index();

    // Marginalize any odometry-based state as well
    for (gtsam::Key k : {V(stateIdx), B(stateIdx), M(stateIdx), S(stateIdx), I(stateIdx)}) {
        if (state_.exists(k)) {
            const auto &slots = variableIndex[k];
            keysToMarg.insert(k);
            facsToMarg.insert(slots.begin(), slots.end());
        }
    }
}

// ------------------------- Public Methods ------------------------- //

void FixedLagBackend::marginalize() {
    // ------------------------- Find variables to marginalize ------------------------- //
    // Figure out which state to marginalize, if any
    gtsam::Key stateToMarg;
    bool isKeyframe;
    bool toMarg = getStateToMarg(stateToMarg, isKeyframe);
    if (!toMarg)
        return;

    gtsam::KeySet keysToMarg;
    gtsam::KeySet keysToDrop;
    FactorSet facsToMarg;
    FactorSet facsToDrop;

    // Figure out which landmarks & odom to marginalize/drop
    const gtsam::VariableIndex variableIndex(graph_);
    getLMMargDrop(stateToMarg, variableIndex, keysToMarg, keysToDrop, facsToMarg, facsToDrop);
    getOdomMargDrop(stateToMarg, variableIndex, keysToMarg, keysToDrop, facsToMarg, facsToDrop);

    // ------------------------- Add any other factors that are in states markov blanket ------------------------- //
    const gtsam::FactorIndices &stateFactorSlots = variableIndex[stateToMarg];
    // If it's a keyframe, we marginalize everything that's not already being dropped
    if (isKeyframe) {
        for (uint64_t slot : stateFactorSlots) {
            if (facsToDrop.find(slot) == facsToDrop.end()) {
                facsToMarg.insert(slot);
            }
        }
    }
    // If not a keyframe, rest will be dropped if stereo factors
    // This will be the stereofactors connected to other states
    else {
        for (uint64_t slot : stateFactorSlots) {
            auto stereo = boost::dynamic_pointer_cast<StereoFactor>(graph_.at(slot));
            if (stereo && facsToMarg.find(slot) == facsToMarg.end()) {
                // std::cout << gtsam::Symbol(stereo->key2()) << std::endl;
                facsToDrop.insert(slot);
            } else {
                facsToMarg.insert(slot);
            }
        }
    }

    // ------------------------- Perform dropping & marginalizing ------------------------- //
    // Drop factors & variables
    for (uint64_t slot : facsToDrop) {
        graph_.remove(slot);
        graphEmptySlots_.push(slot);
    }
    for (gtsam::Key key : keysToDrop) {
        state_.erase(key);
    }

    // Remove factors to marginalize from original graph & move to a new graph
    gtsam::NonlinearFactorGraph removedFactors;
    for (uint64_t slot : facsToMarg) {
        removedFactors.push_back(graph_.at(slot));
        graph_.remove(slot);
        graphEmptySlots_.push(slot);
    }

    // Combine all keys to marginalize together
    gtsam::KeyVector keysToMargVec(keysToMarg.begin(), keysToMarg.end());
    keysToMargVec.push_back(stateToMarg);

    // Make marginal prior & insert
    const gtsam::GaussianFactorGraph::shared_ptr linearFactorGraph = removedFactors.linearize(state_);
    const gtsam::GaussianFactorGraph::shared_ptr marginalLinear =
        linearFactorGraph->eliminatePartialMultifrontal(keysToMargVec).second;
    gtsam::NonlinearFactorGraph marginalFactors =
        gtsam::LinearContainerFactor::ConvertLinearGraph(*marginalLinear, state_);
    marginalIdx_ = graph_.size();
    graph_.add(marginalFactors);

    for (gtsam::Key key : keysToMargVec) {
        state_.erase(key);
    }
}

void FixedLagBackend::addMeasurements(gtsam::NonlinearFactorGraph graph, uint64_t stateIdx) {
    gtsam::KeySet allKeys;
    regFrames_.push(stateIdx);

    for (auto &factor : graph) {
        // Try cast to all types of factors we use
        auto preint = boost::dynamic_pointer_cast<gtsam::CombinedImuFactor>(factor);
        auto stereo = boost::dynamic_pointer_cast<StereoFactor>(factor);
        auto wheel5 = boost::dynamic_pointer_cast<rose::WheelFactor5>(factor);
        auto wheel4intr = boost::dynamic_pointer_cast<rose::WheelFactor4Intrinsics>(factor);
        auto wheel3 = boost::dynamic_pointer_cast<rose::WheelFactor3>(factor);
        auto wheel2 = boost::dynamic_pointer_cast<rose::WheelFactor2>(factor);
        auto priorX = boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Pose3>>(factor);
        auto priorV = boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::Velocity3>>(factor);
        auto priorB = boost::dynamic_pointer_cast<gtsam::PriorFactor<gtsam::imuBias::ConstantBias>>(factor);
        auto noise_model = boost::dynamic_pointer_cast<gtsam::NoiseModelFactor>(factor);

        // Compute any estimate that we need from the factor
        if (preint) {
            addIMUEstimate(preint);
        } else if (stereo) {
            addLandmarkEstimate(stereo);
        } else if (wheel5) {
            addWheel5Estimate(wheel5);
        } else if (wheel4intr) {
            addWheel4IntrEstimate(wheel4intr);
        } else if (wheel3) {
            addWheel3Estimate(wheel3);
        } else if (wheel2) {
            addWheel2Estimate(wheel2);
        } else if (priorX) {
            addPriorEstimate(priorX);
        } else if (priorV) {
            addPriorEstimate(priorV);
        } else if (priorB) {
            addPriorEstimate(priorB);
        }

        if (stereo) {
            gtsam::Key key1 = stereo->key1();
            gtsam::Key key2 = stereo->key2();
            if (mapKey2Key_.find(key1) == mapKey2Key_.end()) {
                mapKey2Key_[key1] = gtsam::KeySet();
            }
            if (mapKey2Key_.find(key2) == mapKey2Key_.end()) {
                mapKey2Key_[key2] = gtsam::KeySet();
            }

            mapKey2Key_[key1].insert(key2);
            mapKey2Key_[key2].insert(key1);
        }

        // Add factor into graph
        noise_model = processFactor(noise_model);
        if (graphEmptySlots_.size() != 0) {
            uint64_t slot = graphEmptySlots_.front();
            graph_.replace(slot, noise_model);
            graphEmptySlots_.pop();
        } else {
            graph_.push_back(factor);
        }
    }
}