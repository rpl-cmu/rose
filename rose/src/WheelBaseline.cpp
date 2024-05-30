#include "rose/WheelBaseline.h"

// ------------------------- Preintegrated Wheel Measurements ------------------------- //
namespace rose {

PreintegratedWheelBaseline::PreintegratedWheelBaseline(const boost::shared_ptr<PreintegratedWheelParams> p) : Base(p) {
    resetIntegration();
}

PreintegratedWheelBaseline::PreintegratedWheelBaseline(Base base) : Base(base) {}

void PreintegratedWheelBaseline::integrateVelocities(double w, double v, double dt) {
    // Find change in states
    double deltaTheta = w * dt;
    double deltaD = v * dt;
    double thetaCov_ = (p_->omegaVelCov(0, 0) / dt) * dt * dt; // -> discrete -> not rates
    double posCov_ = (p_->omegaVelCov(1, 1) / dt) * dt * dt;   // -> discrete -> not rates

    gtsam::Vector2 deltaP;
    deltaP << deltaD * std::cos(deltaTheta / 2), deltaD * std::sin(deltaTheta / 2);

    // Preintegrate states
    gtsam::Rot2 deltaRik(preint_[0]);
    // Since preint is all in SE(2) can simply add thetas
    preint_[0] += deltaTheta;
    preint_.segment<2>(1) += deltaRik * deltaP;

    // Find all gradients for covariance propagation
    gtsam::Vector2 dPdd;
    dPdd << std::cos(deltaTheta / 2), std::sin(deltaTheta / 2);
    gtsam::Vector2 dPdTheta;
    dPdTheta << -deltaD * std::sin(deltaTheta / 2), deltaD * std::cos(deltaTheta / 2);
    gtsam::Vector2 Pcross;
    Pcross << deltaP[1], -deltaP[0];

    // Update covariance
    preintMeasCov_.block<2, 2>(1, 1) += dPdd * dPdd.transpose() * posCov_ +
                                        dPdTheta * dPdTheta.transpose() * thetaCov_ +
                                        Pcross * Pcross.transpose() * preintMeasCov_(0, 0);
    preintMeasCov_(0, 0) += thetaCov_;

    // Handle degeneracy in case of no movement
    if (preintMeasCov_(2, 2) <= 1e-8) {
        preintMeasCov_(2, 2) = 1e-8;
    }
    // preintMeasCov_.block<3,3>(0,0) = gtsam::I_3x3 * 1e-3;

    deltaTij_ += dt;
}

gtsam::Pose3 PreintegratedWheelBaseline::predict(const gtsam::Pose3 &x_i, boost::optional<gtsam::Matrix &> H1) const {
    gtsam::Pose3 delta(gtsam::Rot3::Rz(preint_[0]), gtsam::Vector3(preint_[1], preint_[2], 0));
    return x_i.compose(delta);
}

gtsam::Vector PreintegratedWheelBaseline::evaluateError(const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j,
                                                        boost::optional<gtsam::Matrix &> H1,
                                                        boost::optional<gtsam::Matrix &> H2) const {
    gtsam::Pose3 measured = pose_i.inverse() * predict(pose_i);
    gtsam::Pose3 pose_ij = pose_i.between(pose_j);

    gtsam::Vector3 error;
    // yaw error
    error[0] = measured.localCoordinates(pose_ij)[2];
    // positional error
    error.tail<2>() = pose_ij.translation().head<2>() - measured.translation().head<2>();

    return error;
}

} // namespace rose