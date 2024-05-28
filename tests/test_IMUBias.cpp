#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/slam/StereoFactor.h>
#include <jrl/Dataset.h>
#include <jrl/DatasetBuilder.h>
#include <jrl/Parser.h>
#include <jrl/Writer.h>
#include <jrl/IOMeasurements.h>
#include <jrl/Types.h>

#include "backend/JRL-custom.h"
#include "backend/JRL.h"

#include "gtest/gtest.h"

using gtsam::symbol_shorthand::X;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::B;

#define EXPECT_MATRICES_EQ(M_actual, M_expected) \
  EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected


TEST(Value, IMUBias){
    // Item to save
    gtsam::imuBias::ConstantBias b(gtsam::Vector3::Constant(6), gtsam::Vector3::Constant(7));

    // Save it 
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values theta;
    jrl::ValueTypes types;

    theta.insert(B(0), b);
    types[B(0)] = jrl_rose::IMUBiasTag;

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {}, {}, jrl::TypedValues(theta, types));

    jrl::Writer writer = makeRoseWriter();
    writer.writeDataset(builder.build(), "imu_bias.jrl");

    // Load it back in!
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("imu_bias.jrl");
    gtsam::imuBias::ConstantBias b_read = dataset.initialization('a').at<gtsam::imuBias::ConstantBias>(B(0));

    // Check to make sure they're the same
    EXPECT_MATRICES_EQ(b.vector(), b_read.vector());
}