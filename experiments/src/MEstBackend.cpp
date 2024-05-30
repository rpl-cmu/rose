#include "MEstBackend.h"

gtsam::NoiseModelFactor::shared_ptr MEstBackend::processFactor(gtsam::NoiseModelFactor::shared_ptr factor) {
    auto stereo = boost::dynamic_pointer_cast<StereoFactor>(factor);
    if (stereo) {
        return factor->cloneWithNewNoiseModel(gtsam::noiseModel::Robust::Create(robustKernel_, factor->noiseModel()));
    } else {
        return factor;
    }
}

void MEstBackend::solve() {
    gtsam::LevenbergMarquardtParams m_params;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, state_, m_params);
    state_ = optimizer.optimize();
}