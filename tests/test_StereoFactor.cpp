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

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

#define EXPECT_MATRICES_EQ(M_actual, M_expected)                                                                       \
    EXPECT_TRUE(M_actual.isApprox(M_expected, 1e-6)) << "  Actual:\n" << M_actual << "\nExpected:\n" << M_expected

typedef typename gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3> StereoFactor;

TEST(Factor, StereoFactor) {
    // Make everything for the stereo factor
    gtsam::Cal3_S2Stereo::shared_ptr m_stereoCalibration =
        boost::make_shared<gtsam::Cal3_S2Stereo>(6, 8, 0.0, 3, 4, 0.1);
    auto m_stereoNoiseModel = gtsam::noiseModel::Isotropic::Sigma(3, 1);
    gtsam::Symbol landmarkKey = L(0);
    gtsam::Symbol poseKey = X(0);
    gtsam::StereoCamera camera(gtsam::Pose3(), m_stereoCalibration);
    gtsam::StereoPoint2 stereoPoint(1, 2, 3);
    gtsam::Point3 estimate = camera.backproject(stereoPoint);

    StereoFactor write_factor(stereoPoint, m_stereoNoiseModel, poseKey, landmarkKey, m_stereoCalibration);

    // Save it
    gtsam::NonlinearFactorGraph graph;
    graph.push_back(write_factor);

    jrl::DatasetBuilder builder("test", {'a'});
    builder.addEntry('a', 0, graph, {jrl_rose::StereoFactorPose3Point3Tag});

    jrl::Writer writer = jrl_rose::makeRoseWriter();
    writer.writeDataset(builder.build(), "stereo.jrl");

    // Load it back in!
    jrl::Parser parser = jrl_rose::makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset("stereo.jrl");
    StereoFactor::shared_ptr read_factor = boost::dynamic_pointer_cast<StereoFactor>(dataset.factorGraph()[0]);

    gtsam::Pose3 x0 = gtsam::Pose3::Identity();
    gtsam::Point3 l0(3, 2, 1);
    EXPECT_TRUE(write_factor.equals(*read_factor));
    EXPECT_MATRICES_EQ(write_factor.evaluateError(x0, l0), read_factor->evaluateError(x0, l0));
}