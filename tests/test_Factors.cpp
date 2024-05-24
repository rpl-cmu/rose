#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include "backend/WheelBaseline.h"
#include "backend/WheelRose.h"

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::M;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

#define EXPECT_MATRICES_EQ(M_actual, M_expected)                                                                       \
    EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-1)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

#define EXPECT_ZERO(v) EXPECT_TRUE(v.isZero(1e-4)) << " Actual is not zero:\n" << v

TEST(Factor, WheelRose) {
    double wl = 0.4;
    double wr = 0.2;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    PreintegratedWheelRose pwm(pwm_params);
    for (int i = 0; i < 10; ++i) {
        pwm.integrateMeasurements(wl, wr, dt);
    }

    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 delta = pwm.predict(gtsam::Pose3::Identity());
    gtsam::Pose3 x1 = b_T_s * delta * b_T_s.inverse();
    gtsam::Pose3 x1_init = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

    WheelFactor2 factor(X(0), X(1), pwm.copy(), b_T_s);

    // Make graph
    gtsam::NonlinearFactorGraph graph;
    graph.addPrior<gtsam::Pose3>(X(0), x0, gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));
    graph.push_back(factor);

    // Make Values
    gtsam::Values values;
    values.insert(X(0), x0);
    values.insert(X(1), x1_init);

    // Solve
    gtsam::LevenbergMarquardtParams params;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    gtsam::Values results = optimizer.optimize();
    gtsam::Pose3 x1_final = results.at<gtsam::Pose3>(X(1));

    EXPECT_ZERO(factor.evaluateError(x0, x1_final));
}

TEST(Factor, WheelFactorDang) {
    double w = 0.1;
    double v = 1;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    PreintegratedWheelBaseline pwm(pwm_params);
    for (int i = 0; i < 1; ++i) {
        pwm.integrateVelocities(w, v, dt);
    }

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = pwm.predict(gtsam::Pose3::Identity());
    gtsam::Pose3 x1_init = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

    WheelFactor2 factor(X(0), X(1), pwm.copy());

    // Make graph
    gtsam::NonlinearFactorGraph graph;
    graph.addPrior<gtsam::Pose3>(X(0), x0, gtsam::noiseModel::Isotropic::Sigma(6, 1e-5));
    graph.push_back(factor);

    // Make Values
    gtsam::Values values;
    values.insert(X(0), x0);
    values.insert(X(1), x1_init);

    // Solve
    gtsam::LevenbergMarquardtParams params;
    gtsam::LevenbergMarquardtOptimizer optimizer(graph, values, params);
    gtsam::Values results = optimizer.optimize();
    gtsam::Pose3 x1_final = results.at<gtsam::Pose3>(X(1));

    EXPECT_ZERO(factor.evaluateError(x0, x1_final));
}