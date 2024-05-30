#include "rose/WheelFactorBase.h"

namespace rose {

// ------------------------- Preintegration Base ------------------------- //
PreintegratedWheelBase::PreintegratedWheelBase(gtsam::Vector6 preint, Eigen::Matrix<double, 12, 12> preintMeasCov,
                                               double deltaTij)
    : preint_(preint), preintMeasCov_(preintMeasCov), deltaTij_(deltaTij) {}

PreintegratedWheelBase::PreintegratedWheelBase(boost::shared_ptr<PreintegratedWheelParams> p) : p_(p) {}

void PreintegratedWheelBase::integrateMeasurements(double wl, double wr, double dt) {
    double baseline = p_->intrinsics[0];
    double radiusL = p_->intrinsics[1];
    double radiusR = p_->intrinsics[2];
    double w = (radiusR * wr - radiusL * wl) / baseline;
    double v = (radiusR * wr + radiusL * wl) / 2;
    integrateVelocities(w, v, dt);
}

// ------------------------- Factor between 2 poses ------------------------- //
WheelFactor2::WheelFactor2(gtsam::Key key1, gtsam::Key key2, PreintegratedWheelBase::shared_ptr pwm,
                           gtsam::Pose3 body_T_sensor)
    : pwm_(pwm), body_T_sensor_(body_T_sensor), Base(gtsam::noiseModel::Gaussian::Covariance(pwm->preintMeasCov().block(
                                                         0, 0, pwm->dimension2(), pwm->dimension2())),
                                                     key1, key2) {}

gtsam::Vector WheelFactor2::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j,
                                          boost::optional<gtsam::Matrix &> H1,
                                          boost::optional<gtsam::Matrix &> H2) const {

    if (H1)
        H1->setZero();
    if (H2)
        H2->setZero();
    gtsam::Matrix H1_comp, H2_comp;
    gtsam::Vector e =
        pwm_->evaluateError(pose_i.compose(body_T_sensor_, H1_comp), pose_j.compose(body_T_sensor_, H2_comp), H1, H2);

    // Combine with body_T_sensor composition
    if (H1 && !H1->isZero()) {
        *H1 *= H1_comp;
    }
    if (H2 && !H2->isZero()) {
        *H2 *= H2_comp;
    }

    // If PWM doesn't implement Jacobians, do them numerically
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &)> errorComputer =
        [this](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j) {
            return pwm_->evaluateError(pose_i.compose(body_T_sensor_), pose_j.compose(body_T_sensor_));
        };

    if (H1 && H1->isZero()) {
        H1->resize(pwm_->dimension2(), 6);
        H1->setZero();
        *H1 = gtsam::numericalDerivative21(errorComputer, pose_i, pose_j);
    }
    if (H2 && H2->isZero()) {
        H2->resize(pwm_->dimension2(), 6);
        H2->setZero();
        *H2 = gtsam::numericalDerivative22(errorComputer, pose_i, pose_j);
    }

    return e;
}

gtsam::Pose3 WheelFactor2::predict(const gtsam::Pose3 &x_i) const {
    gtsam::Pose3 predict = pwm_->predict(x_i.compose(body_T_sensor_));
    return predict * body_T_sensor_.inverse();
}

void WheelFactor2::print(const std::string &s, const gtsam::KeyFormatter &keyFormatter) const {
    std::cout << s << "WheelFactor2(" << keyFormatter(this->key1()) << "," << keyFormatter(this->key2()) << ")\n"
              << " measured: " << pwm_->preint().transpose() << "\n";
    this->noiseModel_->print("  noise model: ");
}

// ------------------------- Factor between 2 poses & slip ------------------------- //
WheelFactor3::WheelFactor3(gtsam::Key key1, gtsam::Key key2, gtsam::Key key3, PreintegratedWheelBase::shared_ptr pwm,
                           gtsam::Pose3 body_T_sensor)
    : pwm_(pwm), body_T_sensor_(body_T_sensor), Base(gtsam::noiseModel::Gaussian::Covariance(pwm->preintMeasCov().block(
                                                         0, 0, pwm->dimension3(), pwm->dimension3())),
                                                     key1, key2, key3) {}

gtsam::Vector WheelFactor3::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j,
                                          const gtsam::Vector2 &slip, boost::optional<gtsam::Matrix &> H1,
                                          boost::optional<gtsam::Matrix &> H2,
                                          boost::optional<gtsam::Matrix &> H3) const {

    if (H1)
        H1->setZero();
    if (H2)
        H2->setZero();
    if (H3)
        H3->setZero();
    gtsam::Matrix H1_comp, H2_comp;
    gtsam::Vector e = pwm_->evaluateError(pose_i.compose(body_T_sensor_, H1_comp), slip,
                                          pose_j.compose(body_T_sensor_, H2_comp), H1, H3, H2);

    // If we got Jacobians out
    // Combine with body_T_sensor_ composition
    if (H1 && !H1->isZero()) {
        *H1 *= H1_comp;
    }
    if (H2 && !H2->isZero()) {
        *H2 *= H2_comp;
    }

    // // If PWM doesn't implement Jacobians, do them numerically
    // std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &, const gtsam::Vector2 &)> errorComputer =
    //     [this](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j, const gtsam::Vector2 &slip) {
    //         return pwm_->evaluateError(pose_i.compose(body_T_sensor_), slip, pose_j.compose(body_T_sensor_));
    //     };

    // if (H1) {
    //     H1->resize(pwm_->dimension2(), 6);
    //     H1->setZero();
    //     *H1 = gtsam::numericalDerivative31(errorComputer, pose_i, pose_j, slip);
    // }
    // if (H2) {
    //     H2->resize(pwm_->dimension2(), 6);
    //     H2->setZero();
    //     *H2 = gtsam::numericalDerivative32(errorComputer, pose_i, pose_j, slip);
    // }
    // if (H3) {
    //     H3->resize(pwm_->dimension2(), 6);
    //     H3->setZero();
    //     *H3 = gtsam::numericalDerivative33(errorComputer, pose_i, pose_j, slip);
    // }

    return e;
}

gtsam::Pose3 WheelFactor3::predict(const gtsam::Pose3 &x_i, const gtsam::Vector2 &slip) const {
    gtsam::Pose3 predict = pwm_->predict(x_i.compose(body_T_sensor_), slip);
    return predict * body_T_sensor_.inverse();
}

void WheelFactor3::print(const std::string &s, const gtsam::KeyFormatter &keyFormatter) const {
    std::cout << s << "WheelFactor3(" << keyFormatter(this->key1()) << "," << keyFormatter(this->key2()) << ","
              << keyFormatter(this->key3()) << ")\n"
              << " measured: " << pwm_->preint().transpose() << "\n";
    this->noiseModel_->print("  noise model: ");
}

// ------------------------- Factor between 2 poses / 2 intrinsics ------------------------- //
WheelFactor4Intrinsics::WheelFactor4Intrinsics(gtsam::Key pose_i, gtsam::Key intr_i, gtsam::Key pose_j,
                                               gtsam::Key intr_j, PreintegratedWheelBase::shared_ptr pwm,
                                               gtsam::Pose3 body_T_sensor)
    : pwm_(pwm), body_T_sensor_(body_T_sensor), Base(gtsam::noiseModel::Gaussian::Covariance(pwm->preintMeasCov().block(
                                                         0, 0, pwm->dimension4(), pwm->dimension4())),
                                                     pose_i, intr_i, pose_j, intr_j) {}

gtsam::Vector WheelFactor4Intrinsics::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                                    const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                                    boost::optional<gtsam::Matrix &> H1,
                                                    boost::optional<gtsam::Matrix &> H2,
                                                    boost::optional<gtsam::Matrix &> H3,
                                                    boost::optional<gtsam::Matrix &> H4) const {

    if (H1)
        H1->setZero();
    if (H2)
        H2->setZero();
    if (H3)
        H3->setZero();
    if (H4)
        H4->setZero();
    gtsam::Matrix H1_comp, H3_comp;
    gtsam::Vector e = pwm_->evaluateError(pose_i.compose(body_T_sensor_, H1_comp), intr_i,
                                          pose_j.compose(body_T_sensor_, H3_comp), intr_j, H1, H2, H3, H4);

    // If we got Jacobians out
    // Combine with body_T_sensor_ composition
    if (H1 && !H1->isZero()) {
        *H1 *= H1_comp;
    }
    if (H3 && !H3->isZero()) {
        *H3 *= H3_comp;
    }

    // std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
    //                             const gtsam::Vector3 &)>
    //     errorComputer = [this](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i, const gtsam::Pose3 &pose_j,
    //                            const gtsam::Vector3 &intr_j) {
    //         return pwm_->evaluateError(pose_i.compose(body_T_sensor_), intr_i, pose_j.compose(body_T_sensor_),
    //         intr_j);
    //     };

    // if (H1) {
    //     H1->resize(pwm_->dimension4(), 6);
    //     H1->setZero();
    //     *H1 = gtsam::numericalDerivative41(errorComputer, pose_i, intr_i, pose_j, intr_j);
    // }
    // if (H2) {
    //     H2->resize(pwm_->dimension4(), 3);
    //     H2->setZero();
    //     *H2 = gtsam::numericalDerivative42(errorComputer, pose_i, intr_i, pose_j, intr_j);
    // }
    // if (H3) {
    //     H3->resize(pwm_->dimension4(), 6);
    //     H3->setZero();
    //     *H3 = gtsam::numericalDerivative43(errorComputer, pose_i, intr_i, pose_j, intr_j);
    // }
    // if (H4) {
    //     H4->resize(pwm_->dimension4(), 3);
    //     H4->setZero();
    //     *H4 = gtsam::numericalDerivative44(errorComputer, pose_i, intr_i, pose_j, intr_j);
    // }

    return e;
}

gtsam::Pose3 WheelFactor4Intrinsics::predict(const gtsam::Pose3 &x_i, const gtsam::Vector3 &intr) const {
    gtsam::Pose3 predict = pwm_->predict(x_i.compose(body_T_sensor_), intr);
    return predict * body_T_sensor_.inverse();
}

void WheelFactor4Intrinsics::print(const std::string &s, const gtsam::KeyFormatter &keyFormatter) const {
    std::cout << s << "WheelFactor4Intrinsics(" << keyFormatter(this->key1()) << "," << keyFormatter(this->key2())
              << "," << keyFormatter(this->key3()) << "," << keyFormatter(this->key4()) << ")\n"
              << " measured: " << pwm_->preint().transpose() << "\n";
    this->noiseModel_->print("  noise model: ");
}

// ------------------------- Factor between 2 poses / 2 intrinsics / slip ------------------------- //
WheelFactor5::WheelFactor5(gtsam::Key pose_i, gtsam::Key intr_i, gtsam::Key pose_j, gtsam::Key intr_j, gtsam::Key slip,
                           PreintegratedWheelBase::shared_ptr pwm, gtsam::Pose3 body_T_sensor)
    : pwm_(pwm), body_T_sensor_(body_T_sensor), Base(gtsam::noiseModel::Gaussian::Covariance(pwm->preintMeasCov().block(
                                                         0, 0, pwm->dimension4(), pwm->dimension5())),
                                                     pose_i, intr_i, pose_j, intr_j, slip) {}

gtsam::Vector WheelFactor5::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                          const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                          const gtsam::Vector2 &slip, boost::optional<gtsam::Matrix &> H1,
                                          boost::optional<gtsam::Matrix &> H2, boost::optional<gtsam::Matrix &> H3,
                                          boost::optional<gtsam::Matrix &> H4,
                                          boost::optional<gtsam::Matrix &> H5) const {
    if (H1)
        H1->setZero();
    if (H2)
        H2->setZero();
    if (H3)
        H3->setZero();
    if (H4)
        H4->setZero();
    if (H5)
        H5->setZero();
    gtsam::Matrix H1_comp, H3_comp;
    gtsam::Vector e = pwm_->evaluateError(pose_i.compose(body_T_sensor_, H1_comp), intr_i,
                                          pose_j.compose(body_T_sensor_, H3_comp), intr_j, slip, H1, H2, H3, H4, H5);

    // If we got Jacobians out
    // Combine with body_T_sensor_ composition
    if (H1 && !H1->isZero()) {
        *H1 *= H1_comp;
    }
    if (H3 && !H3->isZero()) {
        *H3 *= H3_comp;
    }

    // std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
    //                             const gtsam::Vector3 &, const gtsam::Vector2 &)>
    //     errorComputer = [this](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i, const gtsam::Pose3 &pose_j,
    //                            const gtsam::Vector3 &intr_j, const gtsam::Vector2 &slip) {
    //         return pwm_->evaluateError(pose_i.compose(body_T_sensor_), intr_i, pose_j.compose(body_T_sensor_),
    //         intr_j,
    //                                    slip);
    //     };

    // if (H1) {
    //     H1->resize(pwm_->dimension4(), 6);
    //     H1->setZero();
    //     *H1 = gtsam::numericalDerivative51(errorComputer, pose_i, intr_i, pose_j, intr_j, slip);
    // }
    // if (H2) {
    //     H2->resize(pwm_->dimension4(), 3);
    //     H2->setZero();
    //     *H2 = gtsam::numericalDerivative52(errorComputer, pose_i, intr_i, pose_j, intr_j, slip);
    // }
    // if (H3) {
    //     H3->resize(pwm_->dimension4(), 6);
    //     H3->setZero();
    //     *H3 = gtsam::numericalDerivative53(errorComputer, pose_i, intr_i, pose_j, intr_j, slip);
    // }
    // if (H4) {
    //     H4->resize(pwm_->dimension4(), 3);
    //     H4->setZero();
    //     *H4 = gtsam::numericalDerivative54(errorComputer, pose_i, intr_i, pose_j, intr_j, slip);
    // }
    // if (H5) {
    //     H5->resize(pwm_->dimension4(), 2);
    //     H5->setZero();
    //     *H5 = gtsam::numericalDerivative55(errorComputer, pose_i, intr_i, pose_j, intr_j, slip);
    // }

    return e;
}

gtsam::Pose3 WheelFactor5::predict(const gtsam::Pose3 &x_i, const gtsam::Vector3 &intr,
                                   const gtsam::Vector2 &slip) const {
    gtsam::Pose3 predict = pwm_->predict(x_i.compose(body_T_sensor_), intr, slip);
    return predict * body_T_sensor_.inverse();
}

void WheelFactor5::print(const std::string &s, const gtsam::KeyFormatter &keyFormatter) const {
    std::cout << s << "WheelFactor5(" << keyFormatter(this->key1()) << "," << keyFormatter(this->key2()) << ","
              << keyFormatter(this->key3()) << "," << keyFormatter(this->key4()) << "," << keyFormatter(this->key4())
              << ")\n"
              << " measured: " << pwm_->preint().transpose() << "\n";
    this->noiseModel_->print("  noise model: ");
}

} // namespace rose