#include "gtsam/geometry/Pose3.h"
#include "gtsam/linear/NoiseModel.h"
#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "jrl/Dataset.h"
#include "jrl/DatasetBuilder.h"

#include "backend/JRL.h"
#include "backend/WheelDang.h"
#include "backend/WheelFactorCov.h"

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::I;
using gtsam::symbol_shorthand::M;
using gtsam::symbol_shorthand::S;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::W;
using gtsam::symbol_shorthand::X;

#define EXPECT_MATRICES_EQ(M_actual, M_expected)                                                                       \
    EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

TEST(JRL, WheelFactorDang) {
    double w = 1;
    double v = 1;
    double dt = 0.1;

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

    // Setup factors
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    PreintegratedWheelDang pwm(pwm_params);
    pwm.integrateVelocities(w, v, dt);
    WheelFactor2 write_factor(X(0), X(1), pwm.copy());

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {WheelDangTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "wheel_dang.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("wheel_dang.jrl");
    WheelFactor2::shared_ptr read_factor = boost::dynamic_pointer_cast<WheelFactor2>(dataset.factorGraph('a')[0]);

    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, x1), read_factor->evaluateError(x0, x1));
}

TEST(JRL, WheelFactorCov) {
    double wl = 1;
    double wr = 1;
    double dt = 0.1;

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});

    // Setup factors
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    PreintegratedWheelCov pwm(pwm_params);
    pwm.integrateMeasurements(wl, wr, dt);
    WheelFactor2 write_factor(X(0), X(1), pwm.copy());

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {WheelCovTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "wheel_cov.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("wheel_cov.jrl");
    WheelFactor2::shared_ptr read_factor = boost::dynamic_pointer_cast<WheelFactor2>(dataset.factorGraph('a')[0]);

    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, x1), read_factor->evaluateError(x0, x1));
}

TEST(JRL, WheelFactorCov3) {
    double wl = 1;
    double wr = 1;
    double dt = 0.1;

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Pose3 b_T_w = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Vector2 slip = gtsam::Vector2::Ones();

    // Setup factors
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    PreintegratedWheelCov pwm(pwm_params);
    pwm.integrateMeasurements(wl, wr, dt);
    WheelFactor3 write_factor(X(0), X(1), W(0), pwm.copy(), b_T_w);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {WheelCovSlipTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "wheel_cov_slip.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("wheel_cov_slip.jrl");
    WheelFactor3::shared_ptr read_factor = boost::dynamic_pointer_cast<WheelFactor3>(dataset.factorGraph('a')[0]);

    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, x1, slip), read_factor->evaluateError(x0, x1, slip));
}

TEST(JRL, WheelFactorCov4) {
    double wl = 1;
    double wr = 1;
    double dt = 0.1;

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Pose3 b_T_w = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Vector3 intr = gtsam::Vector3::Ones();

    // Setup factors
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    PreintegratedWheelCov pwm(pwm_params);
    pwm.integrateMeasurements(wl, wr, dt);
    WheelFactor4Intrinsics write_factor(X(0), I(0), X(1), I(1), pwm.copy(), b_T_w);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {WheelCovIntrTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "wheel_cov_intr.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("wheel_cov_intr.jrl");
    WheelFactor4Intrinsics::shared_ptr read_factor =
        boost::dynamic_pointer_cast<WheelFactor4Intrinsics>(dataset.factorGraph('a')[0]);
    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, intr, x1, intr), read_factor->evaluateError(x0, intr, x1, intr));
}

TEST(JRL, WheelFactorCov5) {
    double wl = 1;
    double wr = 1;
    double dt = 0.1;

    // Setup states
    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Pose3 x1 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    gtsam::Pose3 b_T_w = gtsam::Pose3(gtsam::Rot3::RzRyRx(4, 5, 6), {4, 5, 6});
    gtsam::Vector3 intr = gtsam::Vector3::Ones();
    gtsam::Vector2 slip = gtsam::Vector2::Zero();

    // Setup factors
    auto pwm_params = PreintegratedWheelParams::MakeShared();
    PreintegratedWheelCov pwm(pwm_params);
    pwm.integrateMeasurements(wl, wr, dt);
    WheelFactor5 write_factor(X(0), I(0), X(1), I(1), S(0), pwm.copy(), b_T_w);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {WheelCovIntrSlipTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "wheel_cov_intr_slip.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("wheel_cov_intr_slip.jrl");
    WheelFactor5::shared_ptr read_factor = boost::dynamic_pointer_cast<WheelFactor5>(dataset.factorGraph('a')[0]);
    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, intr, x1, intr, slip),
                       read_factor->evaluateError(x0, intr, x1, intr, slip));
}

TEST(JRL, PlanarPriorFactor) {
    // make factor
    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    PlanarPriorFactor write_factor(X(0), gtsam::I_2x2, b_T_s);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {PlanarPriorTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "planar_prior.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("planar_prior.jrl");
    PlanarPriorFactor::shared_ptr read_factor =
        boost::dynamic_pointer_cast<PlanarPriorFactor>(dataset.factorGraph('a')[0]);

    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0), read_factor->evaluateError(x0));
}

TEST(JRL, ZPriorFactor) {
    // make factor
    gtsam::Pose3 b_T_s = gtsam::Pose3(gtsam::Rot3(0, 1, 0, 0), {1, 2, 3});
    gtsam::Pose3 x0 = gtsam::Pose3(gtsam::Rot3::RzRyRx(1, 2, 3), {1, 2, 3});
    ZPriorFactor write_factor(X(0), gtsam::I_1x1, b_T_s);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {ZPriorTag});

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "z_prior.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("z_prior.jrl");
    ZPriorFactor::shared_ptr read_factor = boost::dynamic_pointer_cast<ZPriorFactor>(dataset.factorGraph('a')[0]);

    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0), read_factor->evaluateError(x0));
}