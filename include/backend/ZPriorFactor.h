/* ----------------------------------------------------------------------------

Modified from ZPriorFactor from gtsam

 * -------------------------------------------------------------------------- */
#pragma once

#include <gtsam/base/Testable.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include "jrl/IOMeasurements.h"
#include "jrl/IOValues.h"

#include <string>

/**
 * A class for a soft prior on any Value type
 * @ingroup SLAM
 */
class ZPriorFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {

  public:
    typedef gtsam::Pose3 T;

  private:
    typedef NoiseModelFactor1<gtsam::Pose3> Base;
    gtsam::Pose3 body_T_sensor_;

  public:
    /// shorthand for a smart pointer to a factor
    typedef typename boost::shared_ptr<ZPriorFactor> shared_ptr;

    /// Typedef to this class
    typedef ZPriorFactor This;

    /** default constructor - only use for serialization */
    ZPriorFactor() {}

    ~ZPriorFactor() override {}

    /** Constructor */
    ZPriorFactor(gtsam::Key key, const gtsam::Matrix1 &covariance,
                 gtsam::Pose3 body_T_sensor = gtsam::Pose3::Identity())
        : Base(gtsam::noiseModel::Gaussian::Covariance(covariance), key), body_T_sensor_(body_T_sensor) {}

    /// @return a deep copy of this factor
    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(gtsam::NonlinearFactor::shared_ptr(new This(*this)));
    }

    /** implement functions needed for Testable */

    /** print */
    void print(const std::string &s,
               const gtsam::KeyFormatter &keyFormatter = gtsam::DefaultKeyFormatter) const override {
        std::cout << s << "ZPriorFactor on " << keyFormatter(this->key()) << "\n";
        if (this->noiseModel_)
            this->noiseModel_->print("  noise model: ");
        else
            std::cout << "no noise model" << std::endl;
    }

    /** equals */
    bool equals(const gtsam::NonlinearFactor &expected, double tol = 1e-9) const override {
        const This *e = dynamic_cast<const This *>(&expected);
        return e != nullptr && Base::equals(*e, tol) &&
               gtsam::traits<gtsam::Pose3>::Equals(body_T_sensor_, e->body_T_sensor_, tol);
    }

    /** implement functions needed to derive from Factor */

    /** vector of errors */
    gtsam::Vector evaluateError(const T &x, boost::optional<gtsam::Matrix &> H = boost::none) const override {
        gtsam::Matrix H_comp;
        gtsam::Pose3 x_sensor = x.compose(body_T_sensor_, H_comp);
        gtsam::Vector e = x_sensor.translation().tail<1>();

        if (H) {
            gtsam::Matrix H_z;
            x_sensor.transformFrom(gtsam::Vector::Zero(3), H_z);

            H->resize(1, 6);
            H->setZero();
            *H = H_z.block<1, 6>(2, 0) * H_comp;
        }

        return e;
    }

    const gtsam::Pose3 &body_T_sensor() const { return body_T_sensor_; }

  private:
    /** Serialization function */
    friend class boost::serialization::access;
    template <class ARCHIVE> void serialize(ARCHIVE &ar, const unsigned int /*version*/) {
        ar &boost::serialization::make_nvp("NoiseModelFactor1", boost::serialization::base_object<Base>(*this));
    }
}; /// namespace gtsam

// ------------------------- JRL Wrapper ------------------------- //
static const std::string ZPriorTag = "ZPrior";

inline gtsam::NonlinearFactor::shared_ptr parseZPriorFactor(const nlohmann::json &input_json) {
    // Get all required fields
    uint64_t key = input_json["key"].get<uint64_t>();
    gtsam::Matrix1 covariance = jrl::io_measurements::parseCovariance(input_json["covariance"], 1);
    gtsam::Pose3 body_T_sensor = jrl::io_values::parse<gtsam::Pose3>(input_json["body_T_sensor"]);

    typename ZPriorFactor::shared_ptr factor = boost::make_shared<ZPriorFactor>(key, covariance, body_T_sensor);
    return factor;
}

inline nlohmann::json serializeZPriorFactor(gtsam::NonlinearFactor::shared_ptr factor) {
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