#include "gtsam/geometry/Pose3.h"
#include "gtsam/inference/Symbol.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/navigation/NavState.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>

#include "rose/PlanarPriorFactor.h"
#include "rose/WheelRose.h"
#include "rose/ZPriorFactor.h"

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::I;
using gtsam::symbol_shorthand::M;
using gtsam::symbol_shorthand::S;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;
using namespace rose;

#define EXPECT_MATRICES_EQ(M_actual, M_expected)                                                                       \
    EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-5)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

TEST(Jacobians, WheelRose) {
    double wl = 0.6;
    double wr = 0.4;
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

    gtsam::Pose3 b_T_s = gtsam::Pose3(); // gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    // x0 = pwm.predict(x0, x0);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &)> errorComputer =
        [pwm, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j) {
            return pwm.evaluateError(pose_i, pose_j);
        };

    gtsam::Matrix H1, H2, H1_num, H2_num;
    gtsam::Vector e = pwm.evaluateError(x0, x1, H1, H2);

    H1_num = gtsam::numericalDerivative21(errorComputer, x0, x1);
    H2_num = gtsam::numericalDerivative22(errorComputer, x0, x1);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
}

TEST(Jacobians, WheelFactorSlip) {
    double wl = 0.6;
    double wr = 0.4;
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

    gtsam::Pose3 b_T_s = gtsam::Pose3(); // gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector2 slip = gtsam::Vector2::Ones() / 100;

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector2 &, const gtsam::Pose3 &)> errorComputer =
        [pwm, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Vector2 &slip, const gtsam::Pose3 &pose_j) {
            return pwm.evaluateError(pose_i, slip, pose_j);
        };

    gtsam::Matrix H1, H2, H3, H1_num, H2_num, H3_num;
    gtsam::Vector e = pwm.evaluateError(x0, slip, x1, H1, H2, H3);

    H1_num = gtsam::numericalDerivative31(errorComputer, x0, slip, x1);
    H2_num = gtsam::numericalDerivative32(errorComputer, x0, slip, x1);
    H3_num = gtsam::numericalDerivative33(errorComputer, x0, slip, x1);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
}

TEST(Jacobians, WheelFactorIntr) {
    double wl = 0.6;
    double wr = 0.4;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    pwm_params->intrinsics = gtsam::Vector3(3, 0.1, 0.1);
    PreintegratedWheelRose pwm(pwm_params);
    for (int i = 0; i < 10; ++i) {
        pwm.integrateMeasurements(wl, wr, dt);
    }

    gtsam::Pose3 b_T_s = gtsam::Pose3(); // gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector3 intr_i = gtsam::Vector3(3.1, 0.2, 0.2);
    gtsam::Vector3 intr_j = pwm_params->intrinsics;
    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &)>
        errorComputer =
            [pwm, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i, const gtsam::Pose3 &pose_j,
                         const gtsam::Vector3 &intr_j) { return pwm.evaluateError(pose_i, intr_i, pose_j, intr_j); };

    gtsam::Matrix H2, H3, H4, H2_num, H3_num, H4_num;
    gtsam::Vector e = pwm.evaluateError(x0, intr_i, x1, intr_j, boost::none, H2, H3, H4);

    H2_num = gtsam::numericalDerivative42(errorComputer, x0, intr_i, x1, intr_j);
    H3_num = gtsam::numericalDerivative43(errorComputer, x0, intr_i, x1, intr_j);
    H4_num = gtsam::numericalDerivative44(errorComputer, x0, intr_i, x1, intr_j);

    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
    EXPECT_MATRICES_EQ(H4, H4_num);
}

TEST(Jacobians, WheelFactorIntrSlip) {
    double wl = 0.6;
    double wr = 0.4;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    pwm_params->intrinsics = gtsam::Vector3(3, 0.1, 0.1);
    PreintegratedWheelRose pwm(pwm_params);
    for (int i = 0; i < 10; ++i) {
        pwm.integrateMeasurements(wl, wr, dt);
    }

    gtsam::Pose3 b_T_s = gtsam::Pose3(); // gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector3 intr_i = gtsam::Vector3(3.1, 0.2, 0.2);
    gtsam::Vector3 intr_j = pwm_params->intrinsics;
    gtsam::Vector2 slip = gtsam::Vector2::Ones() / 100;
    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &, const gtsam::Vector2 &)>
        errorComputer = [pwm, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                     const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                     const gtsam::Vector2 &slip) {
            return pwm.evaluateError(pose_i, intr_i, pose_j, intr_j, slip);
        };

    gtsam::Matrix H1, H2, H3, H4, H5, H1_num, H2_num, H3_num, H4_num, H5_num;
    gtsam::Vector e = pwm.evaluateError(x0, intr_i, x1, intr_j, slip, H1, H2, H3, H4, H5);

    H1_num = gtsam::numericalDerivative51(errorComputer, x0, intr_i, x1, intr_j, slip);
    H2_num = gtsam::numericalDerivative52(errorComputer, x0, intr_i, x1, intr_j, slip);
    H3_num = gtsam::numericalDerivative53(errorComputer, x0, intr_i, x1, intr_j, slip);
    H4_num = gtsam::numericalDerivative54(errorComputer, x0, intr_i, x1, intr_j, slip);
    H5_num = gtsam::numericalDerivative55(errorComputer, x0, intr_i, x1, intr_j, slip);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
    EXPECT_MATRICES_EQ(H4, H4_num);
    EXPECT_MATRICES_EQ(H5, H5_num);
}

TEST(Jacobians, Wheel2Factor_Cov) {
    double wl = 0.6;
    double wr = 0.4;
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
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    WheelFactor2 factor(X(0), X(1), pwm.copy(), b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &)> errorComputer =
        [factor, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j) {
            return factor.evaluateError(pose_i, pose_j);
        };

    gtsam::Matrix H1, H2, H1_num, H2_num;
    gtsam::Vector e = factor.evaluateError(x0, x1, H1, H2);

    H1_num = gtsam::numericalDerivative21(errorComputer, x0, x1);
    H2_num = gtsam::numericalDerivative22(errorComputer, x0, x1);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
}

TEST(Jacobians, Wheel3Factor_Cov) {
    double wl = 0.6;
    double wr = 0.4;
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
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector2 slip = gtsam::Vector2::Zero();
    WheelFactor3 factor(X(0), X(1), S(0), pwm.copy(), b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Pose3 &, const gtsam::Vector2 &)> errorComputer =
        [factor, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Pose3 &pose_j, const gtsam::Vector2 &slip) {
            return factor.evaluateError(pose_i, pose_j, slip);
        };

    gtsam::Matrix H1, H2, H3, H1_num, H2_num, H3_num;
    gtsam::Vector e = factor.evaluateError(x0, x1, slip, H1, H2, H3);

    H1_num = gtsam::numericalDerivative31(errorComputer, x0, x1, slip);
    H2_num = gtsam::numericalDerivative32(errorComputer, x0, x1, slip);
    H3_num = gtsam::numericalDerivative33(errorComputer, x0, x1, slip);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
}

TEST(Jacobians, Wheel4Factor_Cov) {
    double wl = 0.6;
    double wr = 0.4;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    pwm_params->intrinsics = gtsam::Vector3(3, 0.1, 0.1);
    PreintegratedWheelRose pwm(pwm_params);
    for (int i = 0; i < 10; ++i) {
        pwm.integrateMeasurements(wl, wr, dt);
    }

    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector3 intr_i = gtsam::Vector3(3.1, 0.2, 0.2);
    gtsam::Vector3 intr_j = pwm_params->intrinsics;
    WheelFactor4Intrinsics factor(X(0), I(0), X(1), I(1), pwm.copy(), b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &)>
        errorComputer = [factor, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                        const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j) {
            return factor.evaluateError(pose_i, intr_i, pose_j, intr_j);
        };

    gtsam::Matrix H1, H2, H3, H4, H1_num, H2_num, H3_num, H4_num;
    gtsam::Vector e = factor.evaluateError(x0, intr_i, x1, intr_j, H1, H2, H3, H4);

    H1_num = gtsam::numericalDerivative41(errorComputer, x0, intr_i, x1, intr_j);
    H2_num = gtsam::numericalDerivative42(errorComputer, x0, intr_i, x1, intr_j);
    H3_num = gtsam::numericalDerivative43(errorComputer, x0, intr_i, x1, intr_j);
    H4_num = gtsam::numericalDerivative44(errorComputer, x0, intr_i, x1, intr_j);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
    EXPECT_MATRICES_EQ(H4, H4_num);
}

TEST(Jacobians, Wheel5Factor_Cov) {
    double wl = 0.6;
    double wr = 0.4;
    double dt = 0.1;

    // Setup pwm
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    pwm_params->setWVCovFromWV(1e-4, 1e-4);
    pwm_params->wxCov = 1e-2;
    pwm_params->wyCov = 1e-2;
    pwm_params->vyCov = 1e-4;
    pwm_params->vzCov = 1e-4;
    pwm_params->intrinsics = gtsam::Vector3(3, 0.1, 0.1);
    PreintegratedWheelRose pwm(pwm_params);
    for (int i = 0; i < 10; ++i) {
        pwm.integrateMeasurements(wl, wr, dt);
    }

    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Vector3 intr_i = gtsam::Vector3(3.1, 0.2, 0.2);
    gtsam::Vector3 intr_j = pwm_params->intrinsics;
    gtsam::Vector2 slip = gtsam::Vector2::Ones() / 100;
    WheelFactor5 factor(X(0), I(0), X(1), I(1), S(0), pwm.copy(), b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &, const gtsam::Vector3 &, const gtsam::Pose3 &,
                                const gtsam::Vector3 &, const gtsam::Vector2 &)>
        errorComputer = [factor, b_T_s](const gtsam::Pose3 &pose_i, const gtsam::Vector3 &intr_i,
                                        const gtsam::Pose3 &pose_j, const gtsam::Vector3 &intr_j,
                                        const gtsam::Vector2 &slip) {
            return factor.evaluateError(pose_i, intr_i, pose_j, intr_j, slip);
        };

    gtsam::Matrix H1, H2, H3, H4, H5, H1_num, H2_num, H3_num, H4_num, H5_num;
    gtsam::Vector e = factor.evaluateError(x0, intr_i, x1, intr_j, slip, H1, H2, H3, H4, H5);

    H1_num = gtsam::numericalDerivative51(errorComputer, x0, intr_i, x1, intr_j, slip);
    H2_num = gtsam::numericalDerivative52(errorComputer, x0, intr_i, x1, intr_j, slip);
    H3_num = gtsam::numericalDerivative53(errorComputer, x0, intr_i, x1, intr_j, slip);
    H4_num = gtsam::numericalDerivative54(errorComputer, x0, intr_i, x1, intr_j, slip);
    H5_num = gtsam::numericalDerivative55(errorComputer, x0, intr_i, x1, intr_j, slip);

    EXPECT_MATRICES_EQ(H1, H1_num);
    EXPECT_MATRICES_EQ(H2, H2_num);
    EXPECT_MATRICES_EQ(H3, H3_num);
    EXPECT_MATRICES_EQ(H4, H4_num);
    EXPECT_MATRICES_EQ(H5, H5_num);
}

TEST(Jacobians, PlanarPriorFactor) {
    // Setup states
    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    PlanarPriorFactor factor(X(0), gtsam::I_2x2, b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &)> errorComputer = [factor](const gtsam::Pose3 &pose_i) {
        return factor.evaluateError(pose_i);
    };

    gtsam::Matrix H1, H1_num;
    gtsam::Vector e = factor.evaluateError(x0, H1);

    H1_num = gtsam::numericalDerivative11(errorComputer, x0);

    EXPECT_MATRICES_EQ(H1, H1_num);
}

TEST(Jacobians, ZPriorFactor) {
    // Setup states
    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    ZPriorFactor factor(X(0), gtsam::I_1x1, b_T_s);

    // Setup lambda
    std::function<gtsam::Vector(const gtsam::Pose3 &)> errorComputer = [factor](const gtsam::Pose3 &pose_i) {
        return factor.evaluateError(pose_i);
    };

    gtsam::Matrix H1, H1_num;
    gtsam::Vector e = factor.evaluateError(x0, H1);

    H1_num = gtsam::numericalDerivative11(errorComputer, x0);

    EXPECT_MATRICES_EQ(H1, H1_num);
}

// TEST(Jacobians, PriorFactor) {
//     // Setup states
//     gtsam::Rot3 prior = gtsam::Rot3::RzRyRx(1, 2, 3.5);
//     gtsam::Rot3 x0 = gtsam::Rot3::RzRyRx(1, 2, 3);
//     gtsam::PriorFactor<gtsam::Rot3> factor(X(0), prior);

//     // Setup lambda
//     std::function<gtsam::Vector(const gtsam::Rot3 &)> errorComputer =
//         [factor](const gtsam::Rot3 &pose_i) {
//             return factor.evaluateError(pose_i);
//         };

//     gtsam::Matrix H1, H1_num;
//     gtsam::Vector e = factor.evaluateError(x0, H1);

//     H1_num = gtsam::numericalDerivative11(errorComputer, x0);

//     EXPECT_MATRICES_EQ(H1, H1_num);
// }