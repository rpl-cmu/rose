#pragma once

#include "FixedLagBackend.h"

class LMBackend : public FixedLagBackend {
  protected:
    gtsam::NoiseModelFactor::shared_ptr processFactor(gtsam::NoiseModelFactor::shared_ptr factor);

  public:
    void solve();
    std::string getName() { return "LM"; }
};