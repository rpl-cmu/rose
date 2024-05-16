#include "frontend/JRLFrontend.h"
// #include <chrono>

// using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;
// void init_time(time_point& start){
//     start = std::chrono::high_resolution_clock::now();
// }
// void step(time_point& start, std::string out){
//     auto stop = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
//     std::cout << out << " " << duration.count() << std::endl;
//     init_time(start);
// }

JRLFrontend::JRLFrontend(boost::shared_ptr<FixedLagBackend> backend) : backend_(backend) {}

jrl::TypedValues JRLFrontend::makeTypedValues(gtsam::Values values) {
    jrl::ValueTypes types;
    for (gtsam::Symbol key : values.keys()) {
        if (key.chr() == 'x') {
            types[key] = jrl::Pose3Tag;
        } else if (key.chr() == 'w') {
            types[key] = jrl::Pose3Tag;
        } else if (key.chr() == 'i') {
            types[key] = jrl::Point3Tag;
        } else if (key.chr() == 'v') {
            types[key] = jrl::Point3Tag;
        } else if (key.chr() == 'b') {
            types[key] = jrl::IMUBiasTag;
        } else if (key.chr() == 'l') {
            types[key] = jrl::Point3Tag;
        } else if (key.chr() == 's') {
            types[key] = jrl::Point2Tag;
        } else {
            throw std::range_error("Trying to assign ValueTags, " + std::to_string(key.chr()) + " lacks a definition.");
        }
    }
    return jrl::TypedValues(values, types);
}

gtsam::Values JRLFrontend::run(jrl::Dataset &dataset, std::vector<std::string> sensors, std::string outfile,
                               int n_steps, bool use_tqdm) {
    std::vector<jrl::Entry> measurements = dataset.measurements('a');
    gtsam::Values gt = dataset.groundTruth('a');

    // Set everything up
    tqdm bar;
    bar.set_theme_basic();
    if (use_tqdm) {
    }
    jrl::Writer writer = makeRobustWriter();
    n_steps = (n_steps == 0 || n_steps > measurements.size()) ? measurements.size() : n_steps;
    gtsam::Values finalValues;

    // Iterate over each timestep
    for (uint64_t i = 0; i < n_steps; ++i) {
        jrl::Entry entry = measurements[i].filter(sensors);

        backend_->addMeasurements(entry.measurements, entry.stamp);
        backend_->solve();
        backend_->marginalize();

        // Print each timestep of results
        if (use_tqdm) {
            auto ate = jrl::metrics::computeATE<gtsam::Pose3>(gt, backend_->getState(), false);
            bar.progress(i, n_steps);
            bar.set_label("#F: " + std::to_string(backend_->getGraph().nrFactors()) +
                          //   ", #RF: " + std::to_string(backend_->getCurrNumRegframes()) +
                          //   ", #KF: " + std::to_string(backend_->getCurrNumKeyframes()) +
                          ", ATEt: " + std::to_string(ate.first)); // + ", ATErot: " + std::to_string(ate.second));
        }
        // Copy for our perfect version
        finalValues.insert_or_assign(backend_->getState());
    }
    if (use_tqdm) {
        bar.finish();
    }

    std::string methodInfo = outfile.substr(outfile.find_last_of("/") + 1);
    jrl::Results finalResults(dataset.name(), methodInfo, {'a'}, {{'a', makeTypedValues(finalValues)}});
    writer.writeResults(finalResults, outfile);

    return finalValues;
}