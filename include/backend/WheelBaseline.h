#pragma once

#include "gtsam/base/numericalDerivative.h"
#include "gtsam/base/types.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot2.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/NonlinearFactor.h"

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"

#include "backend/WheelFactorBase.h"

// ------------------------- Wheel Factors ------------------------- //

// TODO: Make equals functions for these classes
class PreintegratedWheelBaseline : public PreintegratedWheelBase {
  public:
    typedef PreintegratedWheelBaseline This;
    typedef PreintegratedWheelBase Base;
    typedef typename boost::shared_ptr<PreintegratedWheelBaseline> shared_ptr;

    PreintegratedWheelBaseline(const boost::shared_ptr<PreintegratedWheelParams> p);
    PreintegratedWheelBaseline(Base base);

    // This are inherited, overriden functions
    void integrateVelocities(double omega, double v, double dt) override;

    size_t dimension2() const override { return 3; }
    gtsam::Pose3 predict(const gtsam::Pose3 &x_i, boost::optional<gtsam::Matrix &> H1 = boost::none) const override;
    gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Pose3 &,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override;

    // Additional functionality
    This::shared_ptr copy() { return boost::make_shared<This>(*this); }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

// ------------------------- JRL WRAPPER ------------------------- //
static const std::string WheelBaselineTag = "WheelBaseline";

inline PreintegratedWheelBase::shared_ptr parsePWMDang(nlohmann::json input_json) {
    PreintegratedWheelBase::shared_ptr pwmBase = parsePWBase(input_json);
    return boost::make_shared<PreintegratedWheelBaseline>(*pwmBase);
}