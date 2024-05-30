#pragma once

#include "FixedLagBackend.h"

class MEstBackend : public FixedLagBackend {
  protected:
    gtsam::noiseModel::mEstimator::Base::shared_ptr robustKernel_;
    std::string kernelName_;

    gtsam::NoiseModelFactor::shared_ptr processFactor(gtsam::NoiseModelFactor::shared_ptr factor);

  public:
    MEstBackend(gtsam::noiseModel::mEstimator::Base::shared_ptr robustKernel, std::string kernelName)
        : robustKernel_(robustKernel), kernelName_(kernelName) {}

    void solve();
    std::string getName() { return "MEst-" + kernelName_; }
};