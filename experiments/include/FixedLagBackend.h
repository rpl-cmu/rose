#pragma once

#include <map>
#include <queue>

// These have to be in a specific order, otherwise we get boost errors
// clang-format off
#include <gtsam/base/Vector.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/base/Value.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/LinearContainerFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
// clang-format on

#include "rose/WheelFactorBase.h"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::I;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::M;
using gtsam::symbol_shorthand::S;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::W;
using gtsam::symbol_shorthand::X;
using StereoFactor = gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>;
using FactorSet = std::set<uint64_t>;

class FixedLagBackend {
  private:
    template <class PriorFactorPointer> void addPriorEstimate(PriorFactorPointer factor);
    void addIMUEstimate(gtsam::CombinedImuFactor::shared_ptr factor);
    void addLandmarkEstimate(StereoFactor::shared_ptr factor);
    void addWheel2Estimate(rose::WheelFactor2::shared_ptr factor);
    void addWheel3Estimate(rose::WheelFactor3::shared_ptr factor);
    void addWheel4Estimate(rose::WheelFactor4::shared_ptr factor);
    void addWheel5Estimate(rose::WheelFactor5::shared_ptr factor);

    bool isDegenerate(gtsam::NonlinearFactor::shared_ptr factor);
    bool getStateToMarg(gtsam::Key &stateToMarg, bool &isKeyframe);
    void getLMMargDrop(gtsam::Key stateToMarg, const gtsam::VariableIndex &variableIndex, gtsam::KeySet &keysToMarg,
                       gtsam::KeySet &keysToDrop, FactorSet &facsToMarg, FactorSet &facsToDrop);
    void getOdomMargDrop(gtsam::Key stateToMarg, const gtsam::VariableIndex &variableIndex, gtsam::KeySet &keysToMarg,
                         gtsam::KeySet &keysToDrop, FactorSet &facsToMarg, FactorSet &facsToDrop);

  protected:
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values state_;

    uint64_t regframeNum_ = 5;
    uint64_t keyframeNum_ = 5;
    double keyframeDist_ = 0.1;

    // Empty slots to fill graph, so graph doesn't grow exponentially
    std::queue<uint64_t> graphEmptySlots_;
    // Connections between states (X) and landmarks (L)
    std::map<uint64_t, gtsam::KeySet> mapKey2Key_;
    // Keyframe indices (NOT GTSAM KEYS)
    std::queue<uint64_t> keyFrames_;
    // Regframe indices (NOT GTSAM KEYS)
    std::queue<uint64_t> regFrames_;

    uint64_t marginalIdx_;

    virtual gtsam::NoiseModelFactor::shared_ptr processFactor(gtsam::NoiseModelFactor::shared_ptr factor) = 0;

  public:
    void setRegframeNum(uint64_t regframeNum) { regframeNum_ = regframeNum; }
    void setKeyframeNum(uint64_t keyframeNum) { keyframeNum_ = keyframeNum; }
    void setKeyframeSpace(float keyframeDist) { keyframeDist_ = keyframeDist; }

    virtual std::string getName() { return "BaseBackend"; }
    gtsam::Values getState() { return state_; };
    gtsam::NonlinearFactorGraph getGraph() { return graph_; };
    uint64_t getCurrNumKeyframes() { return keyFrames_.size(); };
    uint64_t getCurrNumRegframes() { return regFrames_.size(); };

    // Each call to this will only remove 1 state
    void marginalize();
    void addMeasurements(gtsam::NonlinearFactorGraph graph, uint64_t stateIdx);
    virtual void solve() = 0;
};