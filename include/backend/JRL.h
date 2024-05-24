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

static const std::string PriorFactorIntrinsicsTag = "PriorFactorIntrinsics";

// ------------------------- Register Custom Options ------------------------- //
inline jrl::Parser makeRoseParser() {
    jrl::Parser parser;

    // clang-format off
    parser.registerMeasurementParser(WheelRoseSlipTag,      [](json input){ return parseWheelFactor3(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelRoseIntrTag,      [](json input){ return parseWheelFactor4Intrinsics(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelRoseIntrSlipTag,  [](json input){ return parseWheelFactor5(input, parsePWMCov); });

    parser.registerMeasurementParser(WheelRoseTag,           [](json input){ return parseWheelFactor2(input, parsePWMCov); });

    parser.registerMeasurementParser(WheelBaselineTag,          [](json input){ return parseWheelFactor2(input, parsePWMDang); });

    parser.registerMeasurementParser(PlanarPriorTag,        parsePlanarPriorFactor);
    parser.registerMeasurementParser(ZPriorTag,             parseZPriorFactor);

    parser.registerMeasurementParser(PriorFactorIntrinsicsTag,    [](const json& input) { return jrl::io_measurements::parsePrior<gtsam::Point3>(&jrl::io_values::parse<gtsam::Point3>, input); });
    // clang-format on

    return parser;
}

inline jrl::Writer makeRoseWriter() {
    jrl::Writer writer;

    // clang-format off
    writer.registerMeasurementSerializer(WheelRoseSlipTag,      [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor3(WheelRoseSlipTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelRoseIntrTag,      [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor4Intrinsics(WheelRoseIntrTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelRoseIntrSlipTag,  [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor5(WheelRoseIntrSlipTag, serializePWMCov, factor); });

    writer.registerMeasurementSerializer(WheelRoseTag,           [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelRoseTag, serializePWMCov, factor); });

    writer.registerMeasurementSerializer(WheelBaselineTag,          [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelBaselineTag, serializePWBase, factor); });
    
    writer.registerMeasurementSerializer(PlanarPriorTag,        serializePlanarPriorFactor);
    writer.registerMeasurementSerializer(ZPriorTag,             serializeZPriorFactor);

    writer.registerMeasurementSerializer(PriorFactorIntrinsicsTag,    [](gtsam::NonlinearFactor::shared_ptr factor) { return jrl::io_measurements::serializePrior<gtsam::Point3>(&jrl::io_values::serialize<gtsam::Point3>, PriorFactorIntrinsicsTag, factor); });
    // clang-format on

    return writer;
}