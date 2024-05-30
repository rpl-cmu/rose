#pragma once

#include "gtsam/base/numericalDerivative.h"
#include "gtsam/base/types.h"
#include "gtsam/geometry/Pose3.h"
#include "gtsam/geometry/Rot2.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/NonlinearFactor.h"

// ------------------------- Preintegration Base ------------------------- //
namespace rose {

class NotImplemented : public std::logic_error {
  public:
    NotImplemented() : std::logic_error("Function not yet implemented"){};
};

struct PreintegratedWheelParams {
    double wxCov = 1;
    double wyCov = 1;
    double vyCov = 1;
    double vzCov = 1;

    gtsam::Matrix2 omegaVelCov = gtsam::I_2x2;

    gtsam::Matrix3 manCov = gtsam::I_3x3;
    gtsam::Matrix3 manInitCov = gtsam::I_3x3;
    double manPosCov = 1;
    double manOrienCov = 1;

    // baseline, radiusL, radiusR
    gtsam::Vector3 intrinsics = gtsam::Vector3::Ones();
    gtsam::Matrix3 intrinsicsBetweenCov = gtsam::Matrix3::Identity();

    static boost::shared_ptr<PreintegratedWheelParams> MakeShared() {
        return boost::shared_ptr<PreintegratedWheelParams>(new PreintegratedWheelParams());
    }

    gtsam::Matrix2 intrinsicsMat() {
        double baseline = intrinsics[0];
        double radiusL = intrinsics[1];
        double radiusR = intrinsics[2];
        gtsam::Matrix2 wheel2body;
        wheel2body << -radiusL / baseline, radiusR / baseline, radiusL / 2, radiusR / 2;
        return wheel2body;
    }

    void setWVCovFromWheel(double wlCov, double wrCov) {
        gtsam::Matrix2 wheel2body = intrinsicsMat();
        gtsam::Matrix2 wlwrCov;
        wlwrCov << wlCov, 0, 0, wrCov;
        omegaVelCov = wheel2body * wlwrCov * wheel2body.transpose();
    }

    void setWVCovFromWV(double wCov, double vCov) { omegaVelCov << wCov, 0, 0, vCov; }

    gtsam::Matrix2 makeFullCov() { return omegaVelCov; }

    gtsam::Matrix6 makeFullVelCov() {
        gtsam::Matrix6 cov = gtsam::Matrix6::Zero();
        cov(0, 0) = wxCov;
        cov(1, 1) = wyCov;
        cov.block<2, 2>(2, 2) = omegaVelCov;
        cov(4, 4) = vyCov;
        cov(5, 5) = vzCov;
        return cov;
    }
};

// TODO: Make equals functions for these classes
class PreintegratedWheelBase {
  protected:
    boost::shared_ptr<PreintegratedWheelParams> p_;
    gtsam::Vector6 preint_;
    Eigen::Matrix<double, 12, 12> preintMeasCov_;
    double deltaTij_;

  public:
    typedef typename boost::shared_ptr<PreintegratedWheelBase> shared_ptr;

    PreintegratedWheelBase(boost::shared_ptr<PreintegratedWheelParams> p);
    PreintegratedWheelBase(gtsam::Vector6 preint, Eigen::Matrix<double, 12, 12> preintMeasCov, double deltaTij);

    boost::shared_ptr<PreintegratedWheelParams> params() const { return p_; }
    gtsam::Vector6 preint() const { return preint_; }
    double deltaTij() const { return deltaTij_; }
    virtual Eigen::Matrix<double, 12, 12> preintMeasCov() const { return preintMeasCov_; }

    // Debatably shoudl be rotated covariance by body_T_sensor adjoint
    // Doesn't seem to make a large difference though
    Eigen::Matrix<double, 12, 12> preintMeasCovAdj(const gtsam::Pose3 &T) {
        gtsam::Matrix6 adj = T.AdjointMap();
        Eigen::Matrix<double, 12, 12> adj_cov = preintMeasCov_;
        adj_cov.block<6, 6>(0, 0) = adj * preintMeasCov_.block<6, 6>(0, 0) * adj.transpose();
        return adj_cov;
    }

    virtual void resetIntegration() {
        preint_.setZero();
        preintMeasCov_.setZero();
        deltaTij_ = 0;
    }

    // In Inherited Classes, implement any/all of these
    virtual void integrateMeasurements(double wl, double wr, double dt);
    virtual void integrateVelocities(double omega, double v, double dt) { throw NotImplemented(); }

    // Used in WheelFactor2
    virtual size_t dimension2() const { throw NotImplemented(); }
    virtual gtsam::Pose3 predict(const gtsam::Pose3 &, boost::optional<gtsam::Matrix &> H1 = boost::none) const {
        throw NotImplemented();
    }
    virtual gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Pose3 &,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none) const {
        throw NotImplemented();
    }

    // Used in WheelFactor3 w/ Slip
    virtual size_t dimension3() const { throw NotImplemented(); }
    virtual gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector2 &,
                                 boost::optional<gtsam::Matrix &> H1 = boost::none,
                                 boost::optional<gtsam::Matrix &> H2 = boost::none) const {
        throw NotImplemented();
    }
    virtual gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector2 &, const gtsam::Pose3 &,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none) const {
        throw NotImplemented();
    }

    // Used in WheelFactor4 w/ Intrinsics
    virtual size_t dimension4() const { throw NotImplemented(); }
    virtual gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &,
                                 boost::optional<gtsam::Matrix &> H1 = boost::none,
                                 boost::optional<gtsam::Matrix &> H2 = boost::none) const {
        throw NotImplemented();
    }
    virtual gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                        const gtsam::Vector3 &, boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none) const {
        throw NotImplemented();
    }

    // Used in WheelFactor5
    virtual size_t dimension5() const { throw NotImplemented(); }
    virtual gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Vector2 &,
                                 boost::optional<gtsam::Matrix &> H1 = boost::none,
                                 boost::optional<gtsam::Matrix &> H2 = boost::none,
                                 boost::optional<gtsam::Matrix &> H3 = boost::none) const {
        throw NotImplemented();
    }
    virtual gtsam::Vector evaluateError(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                        const gtsam::Vector3 &, const gtsam::Vector2 &,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none,
                                        boost::optional<gtsam::Matrix &> = boost::none) const {
        throw NotImplemented();
    }

    GTSAM_MAKE_ALIGNED_OPERATOR_NEW
};

// ------------------------- Factors ------------------------- //
class WheelFactor2 : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
  private:
    typedef WheelFactor2 This;
    typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Base;

    PreintegratedWheelBase::shared_ptr pwm_;
    gtsam::Pose3 body_T_sensor_;

  public:
    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<WheelFactor2> shared_ptr;

    WheelFactor2(gtsam::Key key1, gtsam::Key key2, PreintegratedWheelBase::shared_ptr pwm,
                 gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity());

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override;

    void print(const std::string &s,
               const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override;

    PreintegratedWheelBase::shared_ptr pwm() const { return pwm_; }
    gtsam::Pose3 body_T_sensor() const { return body_T_sensor_; }
    gtsam::Pose3 predict(const gtsam::Pose3 &) const;
};

class WheelFactor3 : public gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2> {
  private:
    typedef WheelFactor3 This;
    typedef gtsam::NoiseModelFactor3<gtsam::Pose3, gtsam::Pose3, gtsam::Vector2> Base;

    PreintegratedWheelBase::shared_ptr pwm_;
    gtsam::Pose3 body_T_sensor_;

  public:
    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<WheelFactor3> shared_ptr;

    WheelFactor3(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, PreintegratedWheelBase::shared_ptr pwm,
                 gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity());

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j, const gtsam::Vector2 &slip,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none,
                                boost::optional<gtsam::Matrix &> H3 = boost::none) const override;

    void print(const std::string &s,
               const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override;

    PreintegratedWheelBase::shared_ptr pwm() const { return pwm_; }
    gtsam::Pose3 body_T_sensor() const { return body_T_sensor_; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector2 &) const;
};

class WheelFactor4 : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
  private:
    PreintegratedWheelBase::shared_ptr pwm_;
    gtsam::Pose3 body_T_sensor_;

    typedef WheelFactor4 This;
    typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> Base;

  public:
    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<WheelFactor4> shared_ptr;

    WheelFactor4(gtsam::Key pose_i, gtsam::Key intr_i, gtsam::Key pose_j, gtsam::Key intr_j,
                 PreintegratedWheelBase::shared_ptr pwm, gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity());

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i, const gtsam::Pose3 &pose_j,
                                const gtsam::Vector3 &intr_j, boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none,
                                boost::optional<gtsam::Matrix &> H3 = boost::none,
                                boost::optional<gtsam::Matrix &> H4 = boost::none) const override;

    void print(const std::string &s,
               const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override;

    PreintegratedWheelBase::shared_ptr pwm() const { return pwm_; }
    gtsam::Pose3 body_T_sensor() const { return body_T_sensor_; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &) const;
};

class WheelFactor5
    : public gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector2> {
  private:
    PreintegratedWheelBase::shared_ptr pwm_;
    gtsam::Pose3 body_T_sensor_;

    typedef WheelFactor5 This;
    typedef gtsam::NoiseModelFactor5<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3, gtsam::Vector2> Base;

  public:
    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<WheelFactor5> shared_ptr;

    WheelFactor5(gtsam::Key pose_i, gtsam::Key intr_i, gtsam::Key pose_j, gtsam::Key intr_j, gtsam::Key slip,
                 PreintegratedWheelBase::shared_ptr pwm, gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity());

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i, const gtsam::Pose3 &pose_j,
                                const gtsam::Vector3 &intr_j, const gtsam::Vector2 &slip,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none,
                                boost::optional<gtsam::Matrix &> H3 = boost::none,
                                boost::optional<gtsam::Matrix &> H4 = boost::none,
                                boost::optional<gtsam::Matrix &> H5 = boost::none) const override;

    void print(const std::string &s,
               const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override;

    PreintegratedWheelBase::shared_ptr pwm() const { return pwm_; }
    gtsam::Pose3 body_T_sensor() const { return body_T_sensor_; }
    gtsam::Pose3 predict(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Vector2 &) const;
};

} // namespace rose