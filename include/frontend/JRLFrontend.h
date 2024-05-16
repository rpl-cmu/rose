#pragma once

#include <jrl/Dataset.h>
#include <jrl/IOValues.h>
#include <jrl/Parser.h>
#include <jrl/Writer.h>

#include "backend/FixedLagBackend.h"
#include "backend/JRL.h"
#include "utils/tqdm.h"

using gtsam::symbol_shorthand::B;
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::V;
using gtsam::symbol_shorthand::X;

class JRLFrontend {
  private:
    boost::shared_ptr<FixedLagBackend> backend_;

    // TODO: Some sort of structure to hold intermediate results
    // - timing
    // - errors
    // - outliers results at various checkpoints

    jrl::TypedValues makeTypedValues(gtsam::Values values);

  public:
    JRLFrontend(boost::shared_ptr<FixedLagBackend> backend);

    gtsam::Values run(jrl::Dataset &dataset, std::vector<std::string> sensors, std::string outfile, int n_steps = 0,
                      bool use_tqdm = true);
};