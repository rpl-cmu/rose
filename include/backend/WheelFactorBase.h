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
#include "nlohmann/json.hpp"

// ------------------------- Preintegration Base ------------------------- //
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

class WheelFactor4Intrinsics
    : public gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> {
  private:
    PreintegratedWheelBase::shared_ptr pwm_;
    gtsam::Pose3 body_T_sensor_;

    typedef WheelFactor4Intrinsics This;
    typedef gtsam::NoiseModelFactor4<gtsam::Pose3, gtsam::Vector3, gtsam::Pose3, gtsam::Vector3> Base;

  public:
    // shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<WheelFactor4Intrinsics> shared_ptr;

    WheelFactor4Intrinsics(gtsam::Key pose_i, gtsam::Key intr_i, gtsam::Key pose_j, gtsam::Key intr_j,
                           PreintegratedWheelBase::shared_ptr pwm,
                           gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity());

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

// ------------------------- JRL WRAPPER ------------------------- //
typedef std::function<PreintegratedWheelBase::shared_ptr(json)> PWParser;
typedef std::function<json(PreintegratedWheelBase::shared_ptr)> PWSerializer;

inline PreintegratedWheelBase::shared_ptr parsePWBase(const nlohmann::json &input_json) {
    json measurement_json = input_json["measurement"];
    json covariance_json = input_json["covariance"];
    json deltaTij_json = input_json["deltaTij"];

    // Construct the factor
    gtsam::Vector6 preint = jrl::io_values::parse<gtsam::Vector>(measurement_json);
    typename Eigen::Matrix<double, 12, 12> covariance = jrl::io_measurements::parseCovariance(covariance_json, 12);
    double deltaTij = deltaTij_json.get<double>();

    return boost::make_shared<PreintegratedWheelBase>(preint, covariance, deltaTij);
}

inline nlohmann::json serializePWBase(PreintegratedWheelBase::shared_ptr pwm) {
    json output;

    output["measurement"] = jrl::io_values::serialize<gtsam::Vector>(pwm->preint());
    output["covariance"] = jrl::io_measurements::serializeCovariance(pwm->preintMeasCov());
    output["deltaTij"] = pwm->deltaTij();

    return output;
}

inline gtsam::NonlinearFactor::shared_ptr parseWheelFactor2(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor2::shared_ptr factor = boost::make_shared<WheelFactor2>(key1, key2, pwm, body_T_sensor);
    return factor;
}

inline nlohmann::json serializeWheelFactor2(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor2::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor2>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys().front();
    output["key2"] = wheelFactor->keys().back();
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

inline gtsam::NonlinearFactor::shared_ptr parseWheelFactor3(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor3::shared_ptr factor = boost::make_shared<WheelFactor3>(key1, key2, key3, pwm, body_T_sensor);
    return factor;
}

inline nlohmann::json serializeWheelFactor3(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor3::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor3>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

inline gtsam::NonlinearFactor::shared_ptr parseWheelFactor4Intrinsics(const nlohmann::json &input_json,
                                                                      PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    uint64_t key4 = input_json["key4"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor4Intrinsics::shared_ptr factor =
        boost::make_shared<WheelFactor4Intrinsics>(key1, key2, key3, key4, pwm, body_T_sensor);
    return factor;
}

inline nlohmann::json serializeWheelFactor4Intrinsics(std::string tag, PWSerializer pwser,
                                                      gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor4Intrinsics::shared_ptr wheelFactor =
        boost::dynamic_pointer_cast<WheelFactor4Intrinsics>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["key4"] = wheelFactor->keys()[3];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

inline gtsam::NonlinearFactor::shared_ptr parseWheelFactor5(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    uint64_t key4 = input_json["key4"].get<uint64_t>();
    uint64_t key5 = input_json["key5"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor5::shared_ptr factor =
        boost::make_shared<WheelFactor5>(key1, key2, key3, key4, key5, pwm, body_T_sensor);
    return factor;
}

inline nlohmann::json serializeWheelFactor5(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor5::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor5>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["key4"] = wheelFactor->keys()[3];
    output["key5"] = wheelFactor->keys()[4];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}