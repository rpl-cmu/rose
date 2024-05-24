#include <Eigen/Core>
#include <boost/program_options.hpp>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <vector>

#include <jrl/Dataset.h>
#include <jrl/DatasetBuilder.h>
#include <jrl/IOMeasurements.h>
#include <jrl/Writer.h>

#include "backend/FixedLagBackend.h"
#include "backend/JRL.h"
#include "backend/LMBackend.h"
#include "backend/MEstBackend.h"
#include "frontend/JRLFrontend.h"

// ------------------------- Handle Commandline Args ------------------------- //
namespace po = boost::program_options;
po::variables_map handle_args(int argc, const char *argv[]) {
    // Define the options
    po::options_description options("Allowed options");
    // clang-format off
    options.add_options()
        ("help,h",                                              "Produce help message")
        ("jrl_file,i", po::value<std::string>()->required(),    "(Required) Path to Json Robot Log file.")
        ("method,m",   po::value<std::string>()->default_value("gm"),    "(Default gm) The name of the method to run (e.g. lm, gm, huber, etc).")
        ("kf",         po::value<uint64_t>()->default_value(5),"(Default 5) Number of keyframes.")
        ("rf",         po::value<uint64_t>()->default_value(0),"(Default 5) Number of regframes.")
        ("spacing",    po::value<double>()->default_value(0),"(Default 0.1) Spatial distance between keyframes.")
        ("factors,f",  po::value<std::vector<std::string>>()->multitoken()->default_value(std::vector<std::string>{}, ""), "(Default all) Which factors/sensors should be ran from bag.")
        ("iter,n",     po::value<uint64_t>()->default_value(0), "(Default all) Number of iterations to run.");
    // clang-format on

    // Parse and return the options
    po::variables_map var_map;
    po::store(po::parse_command_line(argc, argv, options), var_map);

    // Handle help special case
    if (var_map.count("help") || argc == 1) {
        std::cout << "run-experiment: Main entry point to run rose methods on JRL datasets. Please provide "
                     "required arguments: "
                  << std::endl;
        std::cout << options << "\n";
        exit(1);
    }

    // Handle all other arguments
    po::notify(var_map);

    return var_map;
}

// ------------------------- Process Estimator type ------------------------- //
boost::shared_ptr<FixedLagBackend> backendFactory(std::string method_name) {
    if (method_name == "lm") {
        return boost::make_shared<LMBackend>();
    } else if (method_name == "huber") {
        return boost::make_shared<MEstBackend>(gtsam::noiseModel::mEstimator::Huber::Create(3), "huber");
    } else if (method_name == "gm") {
        return boost::make_shared<MEstBackend>(gtsam::noiseModel::mEstimator::GemanMcClure::Create(3), "gm");
    } else if (method_name == "tukey") {
        return boost::make_shared<MEstBackend>(gtsam::noiseModel::mEstimator::Tukey::Create(3), "tukey");
    } else {
        throw std::runtime_error("Unknown Method Name");
    }
}

// ------------------------- Process Sensors to include ------------------------- //
std::set<std::string> parseFactors(std::vector<std::string> sensors) {
    std::set<std::string> outSensors{jrl::PriorFactorPose3Tag};

    if (sensors.size() == 0) {
        outSensors.insert(jrl::StereoFactorPose3Point3Tag);
        outSensors.insert(jrl::BetweenFactorPose3Tag);
        outSensors.insert(jrl::CombinedIMUTag);
        outSensors.insert(jrl::PriorFactorPoint3Tag);
        outSensors.insert(jrl::PriorFactorIMUBiasTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "imu") != sensors.end()) {
        outSensors.insert(jrl::CombinedIMUTag);
        outSensors.insert(jrl::PriorFactorPoint3Tag);
        outSensors.insert(jrl::PriorFactorIMUBiasTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "cam") != sensors.end()) {
        outSensors.insert(jrl::StereoFactorPose3Point3Tag);
    }
    if (std::find(sensors.begin(), sensors.end(), "wheel_dang") != sensors.end()) {
        outSensors.insert(WheelDangTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "wheel_cov") != sensors.end()) {
        outSensors.insert(WheelCovTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "wheel_intr") != sensors.end()) {
        outSensors.insert(WheelCovIntrTag);
        outSensors.insert(jrl::PriorFactorPoint3Tag);
    }
    if (std::find(sensors.begin(), sensors.end(), "intr_prior") != sensors.end()) {
        outSensors.insert(PriorFactorIntrinsicsTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "wheel_slip") != sensors.end()) {
        outSensors.insert(WheelCovSlipTag);
        outSensors.insert(jrl::PriorFactorPoint2Tag);
    }
    if (std::find(sensors.begin(), sensors.end(), "wheel_intr_slip") != sensors.end()) {
        outSensors.insert(WheelCovIntrSlipTag);
        outSensors.insert(jrl::PriorFactorPoint2Tag);
        outSensors.insert(jrl::PriorFactorPoint3Tag);
    }
    if (std::find(sensors.begin(), sensors.end(), "quad") != sensors.end()) {
        outSensors.insert(jrl::BetweenFactorPose3Tag);
    }
    if (std::find(sensors.begin(), sensors.end(), "planar_prior") != sensors.end()) {
        outSensors.insert(PlanarPriorTag);
    }
    if (std::find(sensors.begin(), sensors.end(), "z_prior") != sensors.end()) {
        outSensors.insert(ZPriorTag);
    }

    return outSensors;
}

std::string join(const std::vector<std::string> &sequence, const std::string &separator) {
    std::string result;
    for (size_t i = 0; i < sequence.size(); ++i)
        result += sequence[i] + ((i != sequence.size() - 1) ? separator : "");
    return result;
}

int main(int argc, const char *argv[]) {
    // Handle arguments
    auto args = handle_args(argc, argv);
    std::string data_path = args["jrl_file"].as<std::string>();
    uint64_t iter = args["iter"].as<uint64_t>();
    std::string method = args["method"].as<std::string>();
    uint64_t kf = args["kf"].as<uint64_t>();
    uint64_t rf = args["rf"].as<uint64_t>();
    double spacing = args["spacing"].as<double>();
    std::vector<std::string> factors = args["factors"].as<std::vector<std::string>>();
    std::sort(factors.begin(), factors.end());

    // Load in data
    bool use_cbor = (data_path.find("cbor") != std::string::npos);
    jrl::Parser parser = makeRoseParser();
    jrl::Dataset dataset = parser.parseDataset(data_path, use_cbor);

    // Make backend
    boost::shared_ptr<FixedLagBackend> backend = backendFactory(method);
    backend->setKeyframeNum(kf);
    backend->setRegframeNum(rf);
    backend->setKeyframeSpace(spacing);

    // Get factor types to include
    std::set<std::string> toInclude = parseFactors(factors);
    std::set<std::string> inDataset = dataset.measurementTypes('a');
    std::vector<std::string> usingSensors;
    std::set_intersection(toInclude.begin(), toInclude.end(), inDataset.begin(), inDataset.end(),
                          std::inserter(usingSensors, usingSensors.begin()));

    // Make file name
    std::string data_folder = data_path.substr(0, data_path.find_last_of("/"));
    std::string data_meta = dataset.name().substr(dataset.name().find_last_of("-") + 1);
    // clang-format off
    std::string out_file = data_folder + "/"
                                    + "f-" + join(factors, ".") + ".jrr";
    // clang-format on

    // Let us know what's running
    std::cout << "Method: " << backend->getName() << "\n";
    std::cout << "Num RF: " << rf << "\n";
    std::cout << "Num KF: " << kf << "\n";
    std::cout << "Space KF: " << spacing << "\n";
    std::cout << "Iterations: " << iter << "\n";
    std::cout << "Data Meta: " << data_meta << "\n";
    std::cout << "Factors: \n";
    for (std::string s : usingSensors) {
        std::cout << "\t- " << s << "\n";
    }
    std::cout << "Saving results to " << out_file << "\n\n";

    // Make frontend & run!
    JRLFrontend frontend(backend);
    frontend.run(dataset, usingSensors, out_file, iter);
}
