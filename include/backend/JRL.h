#pragma once

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"
#include "jrl/Parser.h"
#include "jrl/Writer.h"

#include "backend/PlanarPriorFactor.h"
#include "backend/WheelBaseline.h"
#include "backend/WheelFactorBase.h"
#include "backend/WheelRose.h"
#include "backend/ZPriorFactor.h"

#include "backend/JRL-custom.h"


// ------------------------- Register Custom Options ------------------------- //
inline jrl::Parser makeRoseParser() {
    jrl::Parser parser;

    // clang-format off
    parser.registerMeasurementParser(WheelRoseTag,         [](const json& input){ return parseWheelFactor2(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelRoseSlipTag,     [](const json& input){ return parseWheelFactor3(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelRoseIntrTag,     [](const json& input){ return parseWheelFactor4Intrinsics(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelRoseIntrSlipTag, [](const json& input){ return parseWheelFactor5(input, parsePWMCov); });

    parser.registerMeasurementParser(WheelBaselineTag,     [](const json& input){ return parseWheelFactor2(input, parsePWMDang); });

    parser.registerMeasurementParser(PlanarPriorTag,       parsePlanarPriorFactor);
    parser.registerMeasurementParser(ZPriorTag,            parseZPriorFactor);

    parser.registerMeasurementParser(jrl_rose::CombinedIMUTag,             [](const json& input){ return jrl_rose::parseCombinedIMUFactor(input); });
    parser.registerMeasurementParser(jrl_rose::StereoFactorPose3Point3Tag, [](const json& input){ return jrl_rose::parseStereoFactor(input); });
    parser.registerMeasurementParser(jrl_rose::PriorFactorIMUBiasTag,      [](const json& input) { return jrl::io_measurements::parsePrior<gtsam::imuBias::ConstantBias>(&jrl_rose::parseIMUBias, input); });
    parser.registerValueParser(jrl_rose::IMUBiasTag,      [](const json& input, gtsam::Key key, gtsam::Values& accum) { return jrl::io_values::valueAccumulator<gtsam::imuBias::ConstantBias>(&jrl_rose::parseIMUBias, input, key, accum); });
    parser.registerValueParser(jrl_rose::StereoPoint2Tag, [](const json& input, gtsam::Key key, gtsam::Values& accum) { return jrl::io_values::valueAccumulator<gtsam::StereoPoint2>(&jrl_rose::parseStereoPoint2, input, key, accum); });
    // clang-format on

    return parser;
}

inline jrl::Writer makeRoseWriter() {
    jrl::Writer writer;

    // clang-format off
    writer.registerMeasurementSerializer(WheelRoseTag,         [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelRoseTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelRoseSlipTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor3(WheelRoseSlipTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelRoseIntrTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor4Intrinsics(WheelRoseIntrTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelRoseIntrSlipTag, [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor5(WheelRoseIntrSlipTag, serializePWMCov, factor); });

    writer.registerMeasurementSerializer(WheelBaselineTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelBaselineTag, serializePWBase, factor); });
    
    writer.registerMeasurementSerializer(PlanarPriorTag,        serializePlanarPriorFactor);
    writer.registerMeasurementSerializer(ZPriorTag,             serializeZPriorFactor);

    writer.registerMeasurementSerializer(jrl_rose::CombinedIMUTag,             [](gtsam::NonlinearFactor::shared_ptr factor) { return jrl_rose::serializeCombinedIMUFactor(jrl_rose::CombinedIMUTag, factor); });
    writer.registerMeasurementSerializer(jrl_rose::StereoFactorPose3Point3Tag, [](gtsam::NonlinearFactor::shared_ptr factor) { return jrl_rose::serializeStereoFactor(jrl_rose::StereoFactorPose3Point3Tag, factor); });
    writer.registerMeasurementSerializer(jrl_rose::PriorFactorIMUBiasTag,         [](gtsam::NonlinearFactor::shared_ptr& factor) { return jrl::io_measurements::serializePrior<gtsam::imuBias::ConstantBias>(&jrl_rose::serializeIMUBias, jrl_rose::PriorFactorIMUBiasTag, factor); });
    writer.registerValueSerializer(jrl_rose::IMUBiasTag,      [](gtsam::Key key, gtsam::Values& vals) { return jrl_rose::serializeIMUBias(vals.at<gtsam::imuBias::ConstantBias>(key)); });
    writer.registerValueSerializer(jrl_rose::StereoPoint2Tag, [](gtsam::Key key, gtsam::Values& vals) { return jrl_rose::serializeStereoPoint2(vals.at<gtsam::StereoPoint2>(key)); });
    // clang-format on

    return writer;
}