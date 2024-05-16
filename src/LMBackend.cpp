#include "backend/LMBackend.h"

gtsam::NoiseModelFactor::shared_ptr LMBackend::processFactor(gtsam::NoiseModelFactor::shared_ptr factor) {
    return factor;
}

void LMBackend::solve() {
    gtsam::LevenbergMarquardtParams m_params;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph_, state_, m_params);
    state_ = optimizer.optimize();
}