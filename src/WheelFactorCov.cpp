#include "backend/WheelFactorCov.h"

// ------------------------- Preintegrated Wheel Measurements ------------------------- //
PreintegratedWheelCov::PreintegratedWheelCov(const boost::shared_ptr<PreintegratedWheelParams> p) : Base(p) {
    resetIntegration();
    intr_est_ = p_->intrinsics;
}

PreintegratedWheelCov::PreintegratedWheelCov(Base base, gtsam::Matrix62 H_slip, gtsam::Matrix63 H_intr,
                                             gtsam::Vector3 intr_est)
    : Base(base), H_slip_(H_slip), H_intr_(H_intr), intr_est_(intr_est) {}

void PreintegratedWheelCov::resetIntegration() {
    H_slip_.setZero();
    H_intr_.setZero();
    Base::resetIntegration();
}

// TODO Reset integration & reset intrinsics

void PreintegratedWheelCov::integrateMeasurements(double wl, double wr, double dt) {
    // Convert angular rates to w, v
    gtsam::Matrix2 T = intrinsicsMat();
    gtsam::Vector2 wl_wr;
    wl_wr << wl, wr;

    gtsam::Matrix62 E = gtsam::Matrix62::Zero();
    E.block<2, 2>(2, 0) = gtsam::Matrix2::Identity();

    // This functor allows for saving computation when exponential map and its
    // derivatives are needed at the same location in so<3>
    const gtsam::Matrix6 Htinv = gtsam::Pose3::ExpmapDerivative(preint_).inverse();

    // Covariance propagation
    // A = partial f / partial x
    std::function<gtsam::Vector6(const gtsam::Vector6 &)> f = [E, T, wl_wr, dt](const gtsam::Vector6 &x) {
        return gtsam::Pose3::ExpmapDerivative(x).inverse() * E * T * wl_wr * dt;
    };
    gtsam::Matrix6 A = gtsam::I_6x6 + gtsam::numericalDerivative11(f, preint_);

    // B = partial f / partial eta
    gtsam::Matrix66 B = Htinv * dt;

    double b = intr_est_[0];
    double rl = intr_est_[1];
    double rr = intr_est_[2];
    gtsam::Matrix23 C;
    C << -(wr * rr - wl * rl) / (b * b), -wl / b, wr / b, 0, wl / 2, wr / 2;

    // Move everything into place
    preint_ = preint_ + Htinv * E * T * wl_wr * dt;
    H_slip_ = A * H_slip_ + Htinv * E * T * dt;
    H_intr_ = A * H_intr_ + Htinv * E * C * dt;

    preintMeasCov_.block<6, 6>(0, 0) =
        A * preintMeasCov_.block<6, 6>(0, 0) * A.transpose() + B * (p_->makeFullVelCov() / dt) * B.transpose();

    preintMeasCov_.block<3, 3>(6, 6) = p_->intrinsicsBetweenCov;

    deltaTij_ += dt;
}

// ------------------------- For WheelFactor2 ------------------------- //
gtsam::Pose3 PreintegratedWheelCov::predict(const gtsam::Pose3 &x_i, boost::optional<gtsam::Matrix &> H1) const {
    gtsam::Vector6 preintCorr = preint_;
    gtsam::Pose3 delta = gtsam::Pose3::Expmap(preint_);
    return x_i.compose(delta, H1);
}

gtsam::Vector PreintegratedWheelCov::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j,
                                                   boost::optional<gtsam::Matrix &> H1,
                                                   boost::optional<gtsam::Matrix &> H2) const {
    gtsam::Pose3 pose_j_est = predict(pose_i, H1);
    gtsam::Matrix H1_comp;
    gtsam::Vector error = pose_j_est.localCoordinates(pose_j, H1_comp, H2);

    if (H1) {
        *H1 = H1_comp * (*H1);
    }
    return error;
}

// ------------------------- For WheelFactor3 ------------------------- //
gtsam::Pose3 PreintegratedWheelCov::predict(const gtsam::Pose3 &x_i, const gtsam::Vector2 &slip,
                                            boost::optional<gtsam::Matrix &> H1,
                                            boost::optional<gtsam::Matrix &> H2) const {

    gtsam::Matrix H_comp, H_exp;

    gtsam::Vector6 preintCorr = preint_ - H_slip_ * slip;
    gtsam::Pose3 delta = gtsam::Pose3::Expmap(preintCorr, H_exp);
    gtsam::Pose3 x_j_hat = x_i.compose(delta, H1, H_comp);

    if (H2) {
        *H2 = -H_comp * H_exp * H_slip_;
    }

    return x_j_hat;
}

gtsam::Vector PreintegratedWheelCov::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector2 &slip,
                                                   const gtsam::Pose3 &pose_j, boost::optional<gtsam::Matrix &> H1,
                                                   boost::optional<gtsam::Matrix &> H2,
                                                   boost::optional<gtsam::Matrix &> H3) const {
    gtsam::Matrix H1_comp;

    gtsam::Pose3 pose_j_est = predict(pose_i, slip, H1, H2);
    gtsam::Vector error = pose_j_est.localCoordinates(pose_j, H1_comp, H3);

    if (H1) {
        *H1 = H1_comp * (*H1);
    }
    if (H2) {
        *H2 = H1_comp * (*H2);
    }

    return error;
}

// ------------------------- For WheelFactor4 ------------------------- //
gtsam::Pose3 PreintegratedWheelCov::predict(const gtsam::Pose3 &x_i, const gtsam::Vector3 &intr,
                                            boost::optional<gtsam::Matrix &> H1,
                                            boost::optional<gtsam::Matrix &> H2) const {
    gtsam::Matrix H_comp, H_exp;

    gtsam::Vector6 preintCorr = preint_ + H_intr_ * (intr - intr_est_);
    gtsam::Pose3 delta = gtsam::Pose3::Expmap(preintCorr, H_exp);
    gtsam::Pose3 x_j_hat = x_i.compose(delta, H1, H_comp);

    if (H2) {
        *H2 = H_comp * H_exp * H_intr_;
    }

    return x_j_hat;
}

gtsam::Vector PreintegratedWheelCov::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                                   const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                                   boost::optional<gtsam::Matrix &> H1,
                                                   boost::optional<gtsam::Matrix &> H2,
                                                   boost::optional<gtsam::Matrix &> H3,
                                                   boost::optional<gtsam::Matrix &> H4) const {
    gtsam::Matrix H1_comp;

    gtsam::Vector9 error;
    gtsam::Pose3 pose_j_est = predict(pose_i, intr_i, H1, H2);
    error.head<6>() = pose_j_est.localCoordinates(pose_j, H1_comp, H3);
    error.tail<3>() = intr_i - intr_j;

    if (H1) {
        *H1 = H1_comp * (*H1);
    }
    if (H2) {
        *H2 = H1_comp * (*H2);
    }

    if (H1) {
        gtsam::Matrix H1_pose = H1.value();
        H1->resize(9, 6);
        H1->setZero();
        H1->block<6, 6>(0, 0) = H1_pose;
    }
    if (H2) {
        gtsam::Matrix H2_pose = H2.value();
        H2->resize(9, 3);
        H2->setZero();
        H2->block<6, 3>(0, 0) = H2_pose;
        H2->block<3, 3>(6, 0) = gtsam::I_3x3;
    }
    if (H3) {
        gtsam::Matrix H3_pose = H3.value();
        H3->resize(9, 6);
        H3->setZero();
        H3->block<6, 6>(0, 0) = H3_pose;
    }
    if (H4) {
        H4->resize(9, 3);
        H4->setZero();
        H4->block<3, 3>(6, 0) = -gtsam::I_3x3;
    }

    return error;
}

// ------------------------- For WheelFactor5 ------------------------- //
gtsam::Pose3 PreintegratedWheelCov::predict(const gtsam::Pose3 &x_i, const gtsam::Vector3 &intr,
                                            const gtsam::Vector2 &slip, boost::optional<gtsam::Matrix &> H1,
                                            boost::optional<gtsam::Matrix &> H2,
                                            boost::optional<gtsam::Matrix &> H3) const {
    gtsam::Matrix H_comp, H_exp;

    gtsam::Vector6 preintCorr = preint_ + H_intr_ * (intr - intr_est_) - H_slip_ * slip;
    gtsam::Pose3 delta = gtsam::Pose3::Expmap(preintCorr, H_exp);
    gtsam::Pose3 x_j_hat = x_i.compose(delta, H1, H_comp);

    if (H2) {
        *H2 = H_comp * H_exp * H_intr_;
    }
    if (H3) {
        *H3 = -H_comp * H_exp * H_slip_;
    }

    return x_j_hat;
}

gtsam::Vector PreintegratedWheelCov::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                                   const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                                   const gtsam::Vector2 &slip, boost::optional<gtsam::Matrix &> H1,
                                                   boost::optional<gtsam::Matrix &> H2,
                                                   boost::optional<gtsam::Matrix &> H3,
                                                   boost::optional<gtsam::Matrix &> H4,
                                                   boost::optional<gtsam::Matrix &> H5) const {
    gtsam::Matrix H1_comp;

    gtsam::Vector9 error;
    gtsam::Pose3 pose_j_est = predict(pose_i, intr_i, slip, H1, H2, H5);
    error.head<6>() = pose_j_est.localCoordinates(pose_j, H1_comp, H3);
    error.tail<3>() = intr_i - intr_j;

    if (H1) {
        *H1 = H1_comp * (*H1);
    }
    if (H2) {
        *H2 = H1_comp * (*H2);
    }
    if (H5) {
        *H5 = H1_comp * (*H5);
    }

    if (H1) {
        gtsam::Matrix H1_pose = H1.value();
        H1->resize(9, 6);
        H1->setZero();
        H1->block<6, 6>(0, 0) = H1_pose;
    }
    if (H2) {
        gtsam::Matrix H2_pose = H2.value();
        H2->resize(9, 3);
        H2->setZero();
        H2->block<6, 3>(0, 0) = H2_pose;
        H2->block<3, 3>(6, 0) = gtsam::I_3x3;
    }
    if (H3) {
        gtsam::Matrix H3_pose = H3.value();
        H3->resize(9, 6);
        H3->setZero();
        H3->block<6, 6>(0, 0) = H3_pose;
    }
    if (H4) {
        H4->resize(9, 3);
        H4->setZero();
        H4->block<3, 3>(6, 0) = -gtsam::I_3x3;
    }
    if (H5) {
        gtsam::Matrix H5_pose = H5.value();
        H5->resize(9, 2);
        H5->setZero();
        H5->block<6, 2>(0, 0) = H5_pose;
    }

    return error;
}