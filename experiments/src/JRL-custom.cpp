#include "JRL-custom.h"

namespace jrl_rose {

// ------------------------- Values ------------------------- //
// StereoPoint2
gtsam::StereoPoint2 parseStereoPoint2(const json& input_json) {
  double uL = input_json["uL"].get<double>();
  double uR = input_json["uR"].get<double>();
  double v = input_json["v"].get<double>();
  return gtsam::StereoPoint2(uL, uR, v);
}

json serializeStereoPoint2(gtsam::StereoPoint2 point) {
  json output;
  output["type"] = StereoPoint2Tag;
  output["uL"] = point.uL();
  output["uR"] = point.uR();
  output["v"] = point.v();
  return output;
}

// ConstantBias
gtsam::imuBias::ConstantBias parseIMUBias(const json& input_json) {
  gtsam::Vector b = jrl::io_values::parse<gtsam::Vector>(input_json);
  return gtsam::imuBias::ConstantBias(b);
}

json serializeIMUBias(gtsam::imuBias::ConstantBias point) {
  json output = jrl::io_values::serialize<gtsam::Vector>(point.vector());
  output["type"] = IMUBiasTag;
  return output;
}

// ------------------------- Matrices ------------------------- //
gtsam::Matrix parseMatrix(const json& input_json, int row, int col) {
  auto v = input_json.get<std::vector<double>>();
  gtsam::Matrix m = Eigen::Map<gtsam::Matrix>(v.data(), row, col);
  return m;
}

json serializeMatrix(gtsam::Matrix mat) {
  std::vector<double> vec(mat.data(), mat.data() + mat.rows() * mat.cols());
  return json(vec);
}

gtsam::Matrix parseCovariance(json input_json, int d) { return parseMatrix(input_json, d, d); }

json serializeCovariance(gtsam::Matrix covariance) { return serializeMatrix(covariance); }

// ------------------------- IMUFactor ------------------------- //
gtsam::NonlinearFactor::shared_ptr parseCombinedIMUFactor(const json& input_json) {
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

json serializeCombinedIMUFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor) {
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
gtsam::NonlinearFactor::shared_ptr parseStereoFactor(const json& input_json) {
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

json serializeStereoFactor(std::string type_tag, gtsam::NonlinearFactor::shared_ptr& factor) {
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

// ------------------------- PlanarPrior ------------------------- //
gtsam::NonlinearFactor::shared_ptr parsePlanarPriorFactor(const nlohmann::json &input_json) {
    // Get all required fields
    uint64_t key = input_json["key"].get<uint64_t>();
    gtsam::Matrix2 covariance = jrl::io_measurements::parseCovariance(input_json["covariance"], 2);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename PlanarPriorFactor::shared_ptr factor =
        boost::make_shared<PlanarPriorFactor>(key, covariance, body_T_sensor);
    return factor;
}

nlohmann::json serializePlanarPriorFactor(gtsam::NonlinearFactor::shared_ptr factor) {
    typename PlanarPriorFactor::shared_ptr ppf = boost::dynamic_pointer_cast<PlanarPriorFactor>(factor);
    gtsam::noiseModel::Gaussian::shared_ptr noise_model =
        boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(ppf->noiseModel());

    nlohmann::json output;
    output["key"] = ppf->keys().front();
    output["covariance"] = jrl::io_measurements::serializeCovariance(noise_model->covariance());
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(ppf->body_T_sensor());
    output["type"] = PlanarPriorTag;

    return output;
}

gtsam::NonlinearFactor::shared_ptr parseZPriorFactor(const nlohmann::json &input_json) {
    // Get all required fields
    uint64_t key = input_json["key"].get<uint64_t>();
    gtsam::Matrix1 covariance = jrl::io_measurements::parseCovariance(input_json["covariance"], 1);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename ZPriorFactor::shared_ptr factor = boost::make_shared<ZPriorFactor>(key, covariance, body_T_sensor);
    return factor;
}

nlohmann::json serializeZPriorFactor(gtsam::NonlinearFactor::shared_ptr factor) {
    typename ZPriorFactor::shared_ptr ppf = boost::dynamic_pointer_cast<ZPriorFactor>(factor);
    gtsam::noiseModel::Gaussian::shared_ptr noise_model =
        boost::dynamic_pointer_cast<gtsam::noiseModel::Gaussian>(ppf->noiseModel());

    nlohmann::json output;
    output["key"] = ppf->keys().front();
    output["covariance"] = jrl::io_measurements::serializeCovariance(noise_model->covariance());
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(ppf->body_T_sensor());
    output["type"] = ZPriorTag;

    return output;
}

// ------------------------- WheelBase ------------------------- //
typedef std::function<PreintegratedWheelBase::shared_ptr(json)> PWParser;
typedef std::function<json(PreintegratedWheelBase::shared_ptr)> PWSerializer;

PreintegratedWheelBase::shared_ptr parsePWBase(const nlohmann::json &input_json) {
    json measurement_json = input_json["measurement"];
    json covariance_json = input_json["covariance"];
    json deltaTij_json = input_json["deltaTij"];

    // Construct the factor
    gtsam::Vector6 preint = jrl::io_values::parse<gtsam::Vector>(measurement_json);
    typename Eigen::Matrix<double, 12, 12> covariance = jrl::io_measurements::parseCovariance(covariance_json, 12);
    double deltaTij = deltaTij_json.get<double>();

    return boost::make_shared<PreintegratedWheelBase>(preint, covariance, deltaTij);
}

nlohmann::json serializePWBase(PreintegratedWheelBase::shared_ptr pwm) {
    json output;

    output["measurement"] = jrl::io_values::serialize<gtsam::Vector>(pwm->preint());
    output["covariance"] = jrl::io_measurements::serializeCovariance(pwm->preintMeasCov());
    output["deltaTij"] = pwm->deltaTij();

    return output;
}

gtsam::NonlinearFactor::shared_ptr parseWheelFactor2(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor2::shared_ptr factor = boost::make_shared<WheelFactor2>(key1, key2, pwm, body_T_sensor);
    return factor;
}

nlohmann::json serializeWheelFactor2(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor2::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor2>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys().front();
    output["key2"] = wheelFactor->keys().back();
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

gtsam::NonlinearFactor::shared_ptr parseWheelFactor3(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor3::shared_ptr factor = boost::make_shared<WheelFactor3>(key1, key2, key3, pwm, body_T_sensor);
    return factor;
}

nlohmann::json serializeWheelFactor3(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor3::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor3>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

gtsam::NonlinearFactor::shared_ptr parseWheelFactor4Intrinsics(const nlohmann::json &input_json,
                                                                      PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    uint64_t key4 = input_json["key4"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor4Intrinsics::shared_ptr factor =
        boost::make_shared<WheelFactor4Intrinsics>(key1, key2, key3, key4, pwm, body_T_sensor);
    return factor;
}

nlohmann::json serializeWheelFactor4Intrinsics(std::string tag, PWSerializer pwser,
                                                      gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor4Intrinsics::shared_ptr wheelFactor =
        boost::dynamic_pointer_cast<WheelFactor4Intrinsics>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["key4"] = wheelFactor->keys()[3];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

gtsam::NonlinearFactor::shared_ptr parseWheelFactor5(const nlohmann::json &input_json, PWParser pwparser) {
    // Get all required fields
    uint64_t key1 = input_json["key1"].get<uint64_t>();
    uint64_t key2 = input_json["key2"].get<uint64_t>();
    uint64_t key3 = input_json["key3"].get<uint64_t>();
    uint64_t key4 = input_json["key4"].get<uint64_t>();
    uint64_t key5 = input_json["key5"].get<uint64_t>();
    PreintegratedWheelBase::shared_ptr pwm = pwparser(input_json);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename WheelFactor5::shared_ptr factor =
        boost::make_shared<WheelFactor5>(key1, key2, key3, key4, key5, pwm, body_T_sensor);
    return factor;
}

nlohmann::json serializeWheelFactor5(std::string tag, PWSerializer pwser,
                                            gtsam::NonlinearFactor::shared_ptr factor) {
    typename WheelFactor5::shared_ptr wheelFactor = boost::dynamic_pointer_cast<WheelFactor5>(factor);
    json output = pwser(wheelFactor->pwm());
    output["key1"] = wheelFactor->keys()[0];
    output["key2"] = wheelFactor->keys()[1];
    output["key3"] = wheelFactor->keys()[2];
    output["key4"] = wheelFactor->keys()[3];
    output["key5"] = wheelFactor->keys()[4];
    output["body_T_sensor"] = jrl::io_values::serialize<gtsam::Pose3>(wheelFactor->body_T_sensor());
    output["type"] = tag;

    return output;
}

// ------------------------- Baseline ------------------------- //
PreintegratedWheelBase::shared_ptr parsePWMBaseline(nlohmann::json input_json) {
    PreintegratedWheelBase::shared_ptr pwmBase = parsePWBase(input_json);
    return boost::make_shared<PreintegratedWheelBaseline>(*pwmBase);
}


// ------------------------- ROSE ------------------------- //
PreintegratedWheelRose::shared_ptr parsePWMRose(const nlohmann::json &input_json) {
    // Get unique things to this factor
    json H_slip_json = input_json["H_slip"];
    gtsam::Matrix62 H_slip = jrl_rose::parseMatrix(H_slip_json, 6, 2);

    json H_intr_json = input_json["H_intr"];
    gtsam::Matrix63 H_intr = jrl_rose::parseMatrix(H_intr_json, 6, 3);

    json intr_est_json = input_json["intr_est"];
    gtsam::Vector3 intr_est = jrl::io_values::parse<gtsam::Vector3>(intr_est_json);

    PreintegratedWheelBase::shared_ptr pwmBase = parsePWBase(input_json);
    return boost::make_shared<PreintegratedWheelRose>(*pwmBase, H_slip, H_intr, intr_est);
}

nlohmann::json serializePWMRose(PreintegratedWheelBase::shared_ptr pwm) {
    json output = serializePWBase(pwm);

    typename PreintegratedWheelRose::shared_ptr pwmRose = boost::dynamic_pointer_cast<PreintegratedWheelRose>(pwm);
    output["H_slip"] = jrl_rose::serializeMatrix(pwmRose->preint_H_slip());
    output["H_intr"] = jrl_rose::serializeMatrix(pwmRose->preint_H_intr());
    output["intr_est"] = jrl::io_values::serialize<gtsam::Vector3>(pwmRose->intr_est());

    return output;
}
}