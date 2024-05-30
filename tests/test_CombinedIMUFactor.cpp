#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/slam/StereoFactor.h>
#include <jrl/Dataset.h>
#include <jrl/DatasetBuilder.h>
#include <jrl/IOMeasurements.h>
#include <jrl/Parser.h>
#include <jrl/Writer.h>

#include "JRL-custom.h"
#include "JRL.h"

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

#define EXPECT_MATRICES_EQ(M_actual, M_expected)                                                                       \
    EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

TEST(Factor, CombinedIMU) {
    // Setup nonstandard params
    boost::shared_ptr<gtsam::PreintegrationCombinedParams> params = gtsam::PreintegrationCombinedParams::MakeSharedU();
    params->accelerometerCovariance = Eigen::Matrix3d::Identity() * 2;
    params->gyroscopeCovariance = Eigen::Matrix3d::Identity() * 3;
    params->biasAccCovariance = Eigen::Matrix3d::Identity() * 4;
    params->biasOmegaCovariance = Eigen::Matrix3d::Identity() * 5;
    params->biasAccOmegaInt = Eigen::Matrix<double, 6, 6>::Identity() * 6;
    params->integrationCovariance = Eigen::Matrix3d::Identity() * 7;

    // Setup measurements
    gtsam::imuBias::ConstantBias b(gtsam::Vector3::Constant(6), gtsam::Vector3::Constant(7));
    gtsam::PreintegratedCombinedMeasurements pim(params, b);
    gtsam::Vector3 accel{1, 2, 3};
    gtsam::Vector3 omega{4, 5, 6};
    double dt = 0.1;
    pim.integrateMeasurement(accel, omega, dt);
    pim.integrateMeasurement(accel, omega, dt);

    // Setup factor
    int i = 0;
    int j = 1;
    gtsam::CombinedImuFactor write_factor(X(i), V(i), X(j), V(j), B(i), B(j), pim);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {jrl_rose::CombinedIMUTag});

    jrl::Writer writer = jrl_rose::makeRoseWriter();
    writer.writeDataset(builder.build(), "combined_imu.jrl");

    // Load it back in!
    jrl::Parser parser = jrl_rose::makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("combined_imu.jrl");
    gtsam::CombinedImuFactor::shared_ptr read_factor =
        boost::dynamic_pointer_cast<gtsam::CombinedImuFactor>(dataset.factorGraph()[0]);

    // Check to make sure they're the same
    EXPECT_TRUE(write_factor.equals(*read_factor));
    gtsam::Pose3 x = gtsam::Pose3::Identity();
    gtsam::Vector3 v(1, 2, 3);
    EXPECT_MATRICES_EQ(write_factor.evaluateError(x, v, x, v, b, b), read_factor->evaluateError(x, v, x, v, b, b));
}