#pragma once

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"
#include "jrl/Metrics.h"

#include "gtsam/base/types.h"
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/slam/StereoFactor.h>

namespace jrl_rose {
static const std::string IMUBiasTag = "IMUBias";
static const std::string StereoPoint2Tag = "StereoPoint2";

static const std::string StereoFactorPose3Point3Tag = "StereoFactorPose3Point3";
static const std::string CombinedIMUTag = "CombinedIMU";
static const std::string PriorFactorIMUBiasTag = "PriorFactorIMUBias";

// ------------------------- Values ------------------------- //
// StereoPoint2
inline gtsam::StereoPoint2 parseStereoPoint2(const json& input_json) {
  double uL = input_json["uL"].get<double>();
  double uR = input_json["uR"].get<double>();
  double v = input_json["v"].get<double>();
  return gtsam::StereoPoint2(uL, uR, v);
}

inline json serializeStereoPoint2(gtsam::StereoPoint2 point) {
  json output;
  output["type"] = StereoPoint2Tag;
  output["uL"] = point.uL();
  output["uR"] = point.uR();
  output["v"] = point.v();
  return output;
}

// ConstantBias
inline gtsam::imuBias::ConstantBias parseIMUBias(const json& input_json) {
  gtsam::Vector b = jrl::io_values::parse<gtsam::Vector>(input_json);
  return gtsam::imuBias::ConstantBias(b);
}

inline json serializeIMUBias(gtsam::imuBias::ConstantBias point) {
  json output = jrl::io_values::serialize<gtsam::Vector>(point.vector());
  output["type"] = IMUBiasTag;
  return output;
}

// ------------------------- Matrices ------------------------- //
inline gtsam::Matrix parseMatrix(const json& input_json, int row, int col) {
  auto v = input_json.get<std::vector<double>>();
  gtsam::Matrix m = Eigen::Map<gtsam::Matrix>(v.data(), row, col);
  return m;
}

inline json serializeMatrix(gtsam::Matrix mat) {
  std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
  return json(vec);
}

inline gtsam::Matrix parseCovariance(json input_json, int d) { return parseMatrix(input_json, d, d); }

inline json serializeCovariance(gtsam::Matrix covariance) { return serializeMatrix(covariance); }

// ------------------------- IMUFactor ------------------------- //
inline gtsam::NonlinearFactor::shared_ptr parseCombinedIMUFactor(const json& input_json) {
  // First construct Params
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> params = gtsam::PreintegrationCombinedParams::MakeSharedU();
  params->accelerometerCovariance = parseCovariance(input_json["accCov"], 3);
  params->gyroscopeCovariance = parseCovariance(input_json["gyroCov"], 3);
  params->biasAccCovariance = parseCovariance(input_json["biasAccCov"], 3);
  params->biasOmegaCovariance = parseCovariance(input_json["biasGyroCov"], 3);
  params->biasAccOmegaInt = parseCovariance(input_json["biasAccOmegaInt"], 6);
  params->integrationCovariance = parseCovariance(input_json["intCov"], 3);
  params->n_gravity = jrl::io_values::parse<gtsam::Vector>(input_json["g"]);

  // Then construct TangentPreintegration
  gtsam::Vector deltaXij = jrl::io_values::parse<gtsam::Vector>(input_json["mm"]);
  gtsam::Matrix H_biasAcc = parseMatrix(input_json["H_biasAcc"], 9, 3);
  gtsam::Matrix H_biasOmega = parseMatrix(input_json["H_biasOmega"], 9, 3);
  double deltaTij = jrl::io_values::parse<double>(input_json["deltaTij"]);
  gtsam::imuBias::ConstantBias biasHat = parseIMUBias(input_json["biasHat"]);
  gtsam::TangentPreintegration tang_pim(params, deltaXij, H_biasAcc, H_biasOmega, biasHat, deltaTij);

  // Now turn it into CombinedPreintegration
  gtsam::Matrix cov = parseCovariance(input_json["cov"], 15);
  gtsam::PreintegratedCombinedMeasurements pim(tang_pim, cov);

  // And finally into a factor
  uint64_t xi = input_json["k0"].get<uint64_t>();
  uint64_t vi = input_json["k1"].get<uint64_t>();
  uint64_t xj = input_json["k2"].get<uint64_t>();
  uint64_t vj = input_json["k3"].get<uint64_t>();
  uint64_t bi = input_json["k4"].get<uint64_t>();
  uint64_t bj = input_json["k5"].get<uint64_t>();
  gtsam::CombinedImuFactor::shared_ptr factor =
      boost::make_shared<gtsam::CombinedImuFactor>(xi, vi, xj, vj, bi, bj, pim);
  return factor;
}

inline json serializeCombinedIMUFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor) {
  json output;
  typename gtsam::CombinedImuFactor::shared_ptr imu_factor =
      boost::dynamic_pointer_cast<gtsam::CombinedImuFactor>(factor);
  gtsam::noiseModel::Gaussian::shared_ptr noise_model =
      boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(imu_factor->noiseModel());
  gtsam::PreintegratedCombinedMeasurements pim = imu_factor->preintegratedMeasurements();

  output["type"] = type_tag;
  for (int i = 0; i < 6; i++) {
    output["k" + std::to_string(i)] = imu_factor->keys()[i];
  }
  output["cov"] = serializeCovariance(noise_model->covariance());
  output["mm"] = jrl::io_values::serialize<gtsam::Vector>(pim.preintegrated());
  output["H_biasAcc"] = serializeMatrix(pim.preintegrated_H_biasAcc());
  output["H_biasOmega"] = serializeMatrix(pim.preintegrated_H_biasOmega());
  output["deltaTij"] = jrl::io_values::serialize(pim.deltaTij());
  output["biasHat"] = serializeIMUBias(pim.biasHat());

  // PreintegrationParams
  boost::shared_ptr<gtsam::PreintegrationCombinedParams> params =
      boost::dynamic_pointer_cast<gtsam::PreintegrationCombinedParams>(pim.params());
  output["accCov"] = serializeCovariance(params->accelerometerCovariance);
  output["gyroCov"] = serializeCovariance(params->gyroscopeCovariance);
  output["biasAccCov"] = serializeCovariance(params->biasAccCovariance);
  output["biasGyroCov"] = serializeCovariance(params->biasOmegaCovariance);
  output["biasAccOmegaInt"] = serializeCovariance(params->biasAccOmegaInt);
  output["intCov"] = serializeCovariance(params->integrationCovariance);
  output["g"] = jrl::io_values::serialize<gtsam::Vector>(params->n_gravity);

  // omegaCoriolis
  // body_P_sensor

  return output;
}

// ------------------------- StereoFactor ------------------------- //
inline gtsam::NonlinearFactor::shared_ptr parseStereoFactor(const json& input_json) {
  // Get all required fields
  json key_pose_json = input_json["kp"];
  json key_landmark_json = input_json["klm"];
  json measurement_json = input_json["mm"];
  json calibration_json = input_json["cal"];
  json covariance_json = input_json["cov"];

  // Get optional field
  boost::optional<gtsam::Pose3> body_T_sensor = boost::none;
  if (input_json.contains("bTs")) {
    body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["bTs"]);
  }

  // Construct the factor
  gtsam::StereoPoint2 measured = parseStereoPoint2(measurement_json);
  gtsam::Vector calibration_vector = parseMatrix(calibration_json, 6, 1);
  gtsam::Cal3_S2Stereo::shared_ptr calibration = boost::make_shared<gtsam::Cal3_S2Stereo>(calibration_vector);

  typename gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>::shared_ptr factor =
      boost::make_shared<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(
          measured, gtsam::noiseModel::Gaussian::Covariance(parseCovariance(covariance_json, 3)),
          key_pose_json.get<uint64_t>(), key_landmark_json.get<uint64_t>(), calibration, body_T_sensor);

  return factor;
}

inline json serializeStereoFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor) {
  json output;
  typename gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>::shared_ptr stereo_factor =
      boost::dynamic_pointer_cast<gtsam::GenericStereoFactor<gtsam::Pose3, gtsam::Point3>>(factor);
  gtsam::noiseModel::Gaussian::shared_ptr noise_model =
      boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(stereo_factor->noiseModel());

  // Usual NoiseModel2 stuff
  output["type"] = type_tag;
  output["kp"] = stereo_factor->keys().front();
  output["klm"] = stereo_factor->keys().back();
  output["cov"] = serializeCovariance(noise_model->covariance());
  output["mm"] = serializeStereoPoint2(stereo_factor->measured());

  // Extra stuff for this factor
  output["cal"] = serializeMatrix(stereo_factor->calibration()->vector());
  boost::optional<gtsam::Pose3> body_T_sensor = stereo_factor->body_P_sensor();
  if (body_T_sensor.is_initialized()) {
    output["bTs"] = jrl::io_values::serialize(body_T_sensor.get());
  }
  return output;
}


// ------------------------- Metrics ------------------------- //
template <class POSE_TYPE>
inline std::pair<double, double> computeATE(gtsam::Values ref, gtsam::Values est, bool align=false, bool align_with_scale=false){
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
