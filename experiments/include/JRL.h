#pragma once

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"
#include "jrl/Parser.h"
#include "jrl/Writer.h"

#include "rose/PlanarPriorFactor.h"
#include "rose/WheelBaseline.h"
#include "rose/WheelFactorBase.h"
#include "rose/WheelRose.h"
#include "rose/ZPriorFactor.h"

#include "JRL-custom.h"


// ------------------------- Register Custom Options ------------------------- //
namespace jrl_rose {
    inline jrl::Parser makeRoseParser() {
        jrl::Parser parser;

        // clang-format off
        parser.registerMeasurementParser(WheelRoseTag,         [](const json& input){ return parseWheelFactor2(input, parsePWMRose); });
        parser.registerMeasurementParser(WheelRoseSlipTag,     [](const json& input){ return parseWheelFactor3(input, parsePWMRose); });
        parser.registerMeasurementParser(WheelRoseIntrTag,     [](const json& input){ return parseWheelFactor4Intrinsics(input, parsePWMRose); });
        parser.registerMeasurementParser(WheelRoseIntrSlipTag, [](const json& input){ return parseWheelFactor5(input, parsePWMRose); });

        parser.registerMeasurementParser(WheelBaselineTag,     [](const json& input){ return parseWheelFactor2(input, parsePWMBaseline); });

        parser.registerMeasurementParser(PlanarPriorTag,       parsePlanarPriorFactor);
        parser.registerMeasurementParser(ZPriorTag,            parseZPriorFactor);

        parser.registerMeasurementParser(CombinedIMUTag,             [](const json& input){ return parseCombinedIMUFactor(input); });
        parser.registerMeasurementParser(StereoFactorPose3Point3Tag, [](const json& input){ return parseStereoFactor(input); });
        parser.registerMeasurementParser(PriorFactorIMUBiasTag,      [](const json& input) { return jrl::io_measurements::parsePrior<gtsam::imuBias::ConstantBias>(&parseIMUBias, input); });
        parser.registerValueParser(IMUBiasTag,      [](const json& input, gtsam::Key key, gtsam::Values& accum) { return jrl::io_values::valueAccumulator<gtsam::imuBias::ConstantBias>(&parseIMUBias, input, key, accum); });
        parser.registerValueParser(StereoPoint2Tag, [](const json& input, gtsam::Key key, gtsam::Values& accum) { return jrl::io_values::valueAccumulator<gtsam::StereoPoint2>(&parseStereoPoint2, input, key, accum); });
        // clang-format on

        return parser;
    }

    inline jrl::Writer makeRoseWriter() {
        jrl::Writer writer;

        // clang-format off
        writer.registerMeasurementSerializer(WheelRoseTag,         [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelRoseTag, serializePWMRose, factor); });
        writer.registerMeasurementSerializer(WheelRoseSlipTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor3(WheelRoseSlipTag, serializePWMRose, factor); });
        writer.registerMeasurementSerializer(WheelRoseIntrTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor4Intrinsics(WheelRoseIntrTag, serializePWMRose, factor); });
        writer.registerMeasurementSerializer(WheelRoseIntrSlipTag, [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor5(WheelRoseIntrSlipTag, serializePWMRose, factor); });

        writer.registerMeasurementSerializer(WheelBaselineTag,     [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelBaselineTag, serializePWBase, factor); });
        
        writer.registerMeasurementSerializer(PlanarPriorTag,        serializePlanarPriorFactor);
        writer.registerMeasurementSerializer(ZPriorTag,             serializeZPriorFactor);

        writer.registerMeasurementSerializer(CombinedIMUTag,             [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeCombinedIMUFactor(CombinedIMUTag, factor); });
        writer.registerMeasurementSerializer(StereoFactorPose3Point3Tag, [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeStereoFactor(StereoFactorPose3Point3Tag, factor); });
        writer.registerMeasurementSerializer(PriorFactorIMUBiasTag,         [](gtsam::NonlinearFactor::shared_ptr& factor) { return jrl::io_measurements::serializePrior<gtsam::imuBias::ConstantBias>(&serializeIMUBias, PriorFactorIMUBiasTag, factor); });
        writer.registerValueSerializer(IMUBiasTag,      [](gtsam::Key key, gtsam::Values& vals) { return serializeIMUBias(vals.at<gtsam::imuBias::ConstantBias>(key)); });
        writer.registerValueSerializer(StereoPoint2Tag, [](gtsam::Key key, gtsam::Values& vals) { return serializeStereoPoint2(vals.at<gtsam::StereoPoint2>(key)); });
        // clang-format on

        return writer;
    }
}