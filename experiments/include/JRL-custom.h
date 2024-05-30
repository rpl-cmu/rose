#pragma once

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"
#include "jrl/Metrics.h"

#include "gtsam/base/types.h"
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/StereoFactor.h>

#include "rose/PlanarPriorFactor.h"
#include "rose/ZPriorFactor.h"
#include "rose/WheelBaseline.h"
#include "rose/WheelFactorBase.h"
#include "rose/WheelRose.h"

using namespace rose;

namespace jrl_rose {

static const std::string IMUBiasTag = "IMUBias";
static const std::string StereoPoint2Tag = "StereoPoint2";

static const std::string StereoFactorPose3Point3Tag = "StereoFactorPose3Point3";
static const std::string CombinedIMUTag = "CombinedIMU";
static const std::string PriorFactorIMUBiasTag = "PriorFactorIMUBias";

static const std::string PlanarPriorTag = "PlanarPrior";
static const std::string ZPriorTag = "ZPrior";
static const std::string WheelBaselineTag = "WheelBaseline";

static const std::string WheelRoseTag = "WheelRose";
static const std::string WheelRoseSlipTag = "WheelRoseSlip";
static const std::string WheelRoseIntrTag = "WheelRoseIntrinsics";
static const std::string WheelRoseIntrSlipTag = "WheelRoseIntrinsicsSlip";

// ------------------------- Values ------------------------- //
// StereoPoint2
gtsam::StereoPoint2 parseStereoPoint2(const json& input_json);

json serializeStereoPoint2(gtsam::StereoPoint2 point);

// ConstantBias
gtsam::imuBias::ConstantBias parseIMUBias(const json& input_json);

json serializeIMUBias(gtsam::imuBias::ConstantBias point);

// ------------------------- Matrices ------------------------- //
gtsam::Matrix parseMatrix(const json& input_json, int row, int col);

json serializeMatrix(gtsam::Matrix mat);

gtsam::Matrix parseCovariance(json input_json, int d);

json serializeCovariance(gtsam::Matrix covariance);

// ------------------------- IMUFactor ------------------------- //
gtsam::NonlinearFactor::shared_ptr parseCombinedIMUFactor(const json& input_json);

json serializeCombinedIMUFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor);

// ------------------------- StereoFactor ------------------------- //
gtsam::NonlinearFactor::shared_ptr parseStereoFactor(const json& input_json);

json serializeStereoFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor);

// ------------------------- PlanarPrior ------------------------- //
gtsam::NonlinearFactor::shared_ptr parsePlanarPriorFactor(const nlohmann::json &input_json);

nlohmann::json serializePlanarPriorFactor(gtsam::NonlinearFactor::shared_ptr factor);

gtsam::NonlinearFactor::shared_ptr parseZPriorFactor(const nlohmann::json &input_json);

nlohmann::json serializeZPriorFactor(gtsam::NonlinearFactor::shared_ptr factor);

// ------------------------- WheelBase ------------------------- //
typedef std::function<PreintegratedWheelBase::shared_ptr(json)> PWParser;
typedef std::function<json(PreintegratedWheelBase::shared_ptr)> PWSerializer;

PreintegratedWheelBase::shared_ptr parsePWBase(const nlohmann::json &input_json);

nlohmann::json serializePWBase(PreintegratedWheelBase::shared_ptr pwm);

gtsam::NonlinearFactor::shared_ptr parseWheelFactor2(const nlohmann::json &input_json, PWParser pwparser);

nlohmann::json serializeWheelFactor2(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor);

gtsam::NonlinearFactor::shared_ptr parseWheelFactor3(const nlohmann::json &input_json, PWParser pwparser);

nlohmann::json serializeWheelFactor3(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor);

gtsam::NonlinearFactor::shared_ptr parseWheelFactor4(const nlohmann::json &input_json,
                                                                      PWParser pwparser);

nlohmann::json serializeWheelFactor4(std::string tag, PWSerializer pwser,
                                                      gtsam::NonlinearFactor::shared_ptr factor);

gtsam::NonlinearFactor::shared_ptr parseWheelFactor5(const nlohmann::json &input_json, PWParser pwparser);

nlohmann::json serializeWheelFactor5(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor);

// ------------------------- Baseline ------------------------- //
PreintegratedWheelBase::shared_ptr parsePWMBaseline(nlohmann::json input_json);

// ------------------------- ROSE ------------------------- //
PreintegratedWheelRose::shared_ptr parsePWMRose(const nlohmann::json &input_json);

nlohmann::json serializePWMRose(PreintegratedWheelBase::shared_ptr pwm);


// ------------------------- Metrics ------------------------- //
template <class POSE_TYPE>
std::pair<double, double> computeATE(gtsam::Values ref, gtsam::Values est, bool align=false, bool align_with_scale=false){
  est = est.filter<POSE_TYPE>();
  if (align) {
    est = jrl::alignment::align<POSE_TYPE>(est, ref, align_with_scale);
  }

  double squared_translation_error = 0.0;
  double squared_rotation_error = 0.0;
  for (auto& key : est.keys()) {
    std::pair<double, double> squared_pose_error =
        jrl::metrics::internal::squaredPoseError<POSE_TYPE>(est.at<POSE_TYPE>(key), ref.at<POSE_TYPE>(key));

    squared_translation_error += squared_pose_error.first;
    squared_rotation_error += squared_pose_error.second;
  }

  // Return the RMSE of the pose errors
  return std::make_pair(std::sqrt(squared_translation_error / est.size()),
                        std::sqrt(squared_rotation_error / est.size()));
}

}
