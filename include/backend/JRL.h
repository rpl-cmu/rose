#pragma once

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"
#include "jrl/Parser.h"
#include "jrl/Writer.h"

#include "backend/PlanarPriorFactor.h"
#include "backend/WheelDang.h"
#include "backend/WheelFactorBase.h"
#include "backend/WheelFactorCov.h"
#include "backend/ZPriorFactor.h"

static const std::string PriorFactorIntrinsicsTag = "PriorFactorIntrinsics";

// ------------------------- Register Custom Options ------------------------- //
inline jrl::Parser makeRobustParser() {
    jrl::Parser parser;

    // clang-format off
    parser.registerMeasurementParser(WheelCovSlipTag,      [](json input){ return parseWheelFactor3(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelCovIntrTag,      [](json input){ return parseWheelFactor4Intrinsics(input, parsePWMCov); });
    parser.registerMeasurementParser(WheelCovIntrSlipTag,  [](json input){ return parseWheelFactor5(input, parsePWMCov); });

    parser.registerMeasurementParser(WheelCovTag,           [](json input){ return parseWheelFactor2(input, parsePWMCov); });

    parser.registerMeasurementParser(WheelDangTag,          [](json input){ return parseWheelFactor2(input, parsePWMDang); });

    parser.registerMeasurementParser(PlanarPriorTag,        parsePlanarPriorFactor);
    parser.registerMeasurementParser(ZPriorTag,             parseZPriorFactor);

    parser.registerMeasurementParser(PriorFactorIntrinsicsTag,    [](const json& input) { return jrl::io_measurements::parsePrior<gtsam::Point3>(&jrl::io_values::parse<gtsam::Point3>, input); });
    // clang-format on

    return parser;
}

inline jrl::Writer makeRobustWriter() {
    jrl::Writer writer;

    // clang-format off
    writer.registerMeasurementSerializer(WheelCovSlipTag,      [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor3(WheelCovSlipTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelCovIntrTag,      [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor4Intrinsics(WheelCovIntrTag, serializePWMCov, factor); });
    writer.registerMeasurementSerializer(WheelCovIntrSlipTag,  [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor5(WheelCovIntrSlipTag, serializePWMCov, factor); });

    writer.registerMeasurementSerializer(WheelCovTag,           [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelCovTag, serializePWMCov, factor); });

    writer.registerMeasurementSerializer(WheelDangTag,          [](gtsam::NonlinearFactor::shared_ptr factor) { return serializeWheelFactor2(WheelDangTag, serializePWBase, factor); });
    
    writer.registerMeasurementSerializer(PlanarPriorTag,        serializePlanarPriorFactor);
    writer.registerMeasurementSerializer(ZPriorTag,             serializeZPriorFactor);

    writer.registerMeasurementSerializer(PriorFactorIntrinsicsTag,    [](gtsam::NonlinearFactor::shared_ptr factor) { return jrl::io_measurements::serializePrior<gtsam::Point3>(&jrl::io_values::serialize<gtsam::Point3>, PriorFactorIntrinsicsTag, factor); });
    // clang-format on

    return writer;
}