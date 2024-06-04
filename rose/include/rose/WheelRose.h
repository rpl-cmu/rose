#pragma once

#include "gtsam/base/numericalDerivative.h"
#include "gtsam/base/types.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot2.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/NonlinearFactor.h"

#include "rose/WheelFactorBase.h"

namespace rose {
// ------------------------- Wheel Factors ------------------------- //

// TODO: Make equals functions for these classes
class PreintegratedWheelRose : public PreintegratedWheelBase {
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
    typedef PreintegratedWheelRose This;
    typedef PreintegratedWheelBase Base;
    typedef typename boost::shared_ptr<PreintegratedWheelRose> shared_ptr;

    PreintegratedWheelRose(const boost::shared_ptr<PreintegratedWheelParams> &p);
    PreintegratedWheelRose(Base base, gtsam::Matrix62 H_slip, gtsam::Matrix63 H_intr, gtsam::Vector3 intr_est);

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

} // namespace rose