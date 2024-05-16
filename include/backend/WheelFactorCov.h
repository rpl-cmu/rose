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
class PreintegratedWheelCov : public PreintegratedWheelBase {
  protected:
    gtsam::Matrix62 H_slip_;
    gtsam::Matrix63 H_intr_;
    gtsam::Vector3 intr_est_;

    gtsam::Matrix2 intrinsicsMat() {
        double baseline = intr_est_[0];
        double radiusL = intr_est_[1];
        double radiusR = intr_est_[2];
        gtsam::Matrix2 wheel2body;
        wheel2body << -radiusL / baseline, radiusR / baseline, radiusL / 2, radiusR / 2;
        return wheel2body;
    }

  public:
    typedef PreintegratedWheelCov This;
    typedef PreintegratedWheelBase Base;
    typedef typename boost::shared_ptr<PreintegratedWheelCov> shared_ptr;

    PreintegratedWheelCov(const boost::shared_ptr<PreintegratedWheelParams> p);
    PreintegratedWheelCov(Base base, gtsam::Matrix62 H_slip, gtsam::Matrix63 H_intr, gtsam::Vector3 intr_est);

    // This are inherited, overriden functions
    void integrateMeasurements(double wl, double wr, double dt) override;
    void resetIntegration();

    size_t dimension2() const override { return 6; }
    gtsam::Pose3 predict(const gtsam::Pose3 &x_i, boost::optional<gtsam::Matrix &> H1 = boost::none) const override;
    gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Pose3 &,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override;

    size_t dimension3() const override { return 6; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector2 &,
                         boost::optional<gtsam::Matrix &> H1 = boost::none,
                         boost::optional<gtsam::Matrix &> H2 = boost::none) const override;
    gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector2 &, const gtsam::Pose3 &,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none) const override;

    size_t dimension4() const override { return 9; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &,
                         boost::optional<gtsam::Matrix &> H1 = boost::none,
                         boost::optional<gtsam::Matrix &> H2 = boost::none) const override;
    gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &, boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none) const override;

    size_t dimension5() const override { return 9; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Vector2 &,
                         boost::optional<gtsam::Matrix &> H1 = boost::none,
                         boost::optional<gtsam::Matrix &> H2 = boost::none,
                         boost::optional<gtsam::Matrix &> H3 = boost::none) const override;
    gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &, const gtsam::Vector2 &,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none,
                                boost::optional<gtsam::Matrix &> = boost::none) const override;

    gtsam::Matrix62 preint_H_slip() const { return H_slip_; }
    gtsam::Matrix63 preint_H_intr() const { return H_intr_; }
    gtsam::Vector3 intr_est() const { return intr_est_; }

    // Additional functionality
    This::shared_ptr copy() { return boost::make_shared<This>(*this); }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

// ------------------------- JRL WRAPPER ------------------------- //
static const std::string WheelCovTag = "WheelCov";
static const std::string WheelCovSlipTag = "WheelCovSlip";
static const std::string WheelCovIntrTag = "WheelCovIntrinsics";
static const std::string WheelCovIntrSlipTag = "WheelCovIntrinsicsSlip";

inline PreintegratedWheelCov::shared_ptr parsePWMCov(const nlohmann::json &input_json) {
    // Get unique things to this factor
    json H_slip_json = input_json["H_slip"];
    gtsam::Matrix62 H_slip = jrl::io_measurements::parseMatrix(H_slip_json, 6, 2);

    json H_intr_json = input_json["H_intr"];
    gtsam::Matrix63 H_intr = jrl::io_measurements::parseMatrix(H_intr_json, 6, 3);

    json intr_est_json = input_json["intr_est"];
    gtsam::Vector3 intr_est = jrl::io_values::parse<gtsam::Vector3>(intr_est_json);

    PreintegratedWheelBase::shared_ptr pwmBase = parsePWBase(input_json);
    return boost::make_shared<PreintegratedWheelCov>(*pwmBase, H_slip, H_intr, intr_est);
}

inline nlohmann::json serializePWMCov(PreintegratedWheelBase::shared_ptr pwm) {
    json output = serializePWBase(pwm);

    typename PreintegratedWheelCov::shared_ptr pwmCov = boost::dynamic_pointer_cast<PreintegratedWheelCov>(pwm);
    output["H_slip"] = jrl::io_measurements::serializeMatrix(pwmCov->preint_H_slip());
    output["H_intr"] = jrl::io_measurements::serializeMatrix(pwmCov->preint_H_intr());
    output["intr_est"] = jrl::io_values::serialize<gtsam::Vector3>(pwmCov->intr_est());

    return output;
}