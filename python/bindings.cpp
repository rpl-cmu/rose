#include <boost/shared_ptr.hpp>

#include <gtsam/linear/LossFunctions.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "backend/JRL.h"
#include "backend/MEstBackend.h"
#include "backend/WheelFactorCov.h"
#include "frontend/JRLFrontend.h"

namespace py = pybind11;
using namespace pybind11::literals;

// Holder type for pybind11
PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE, boost::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);
PYBIND11_DECLARE_HOLDER_TYPE(TYPE_PLACEHOLDER_DONOTUSE, std::shared_ptr<TYPE_PLACEHOLDER_DONOTUSE>);

PYBIND11_MODULE(robust_python, m) {
    // Import gtsam to ensure that python has access to return types
    py::module gtsam = py::module::import("gtsam");
    py::module jrl = py::module::import("jrl");

    // ------------------------- Custom JRL bindings for the below ------------------------- //
    m.attr("WheelCovTag") = py::str(WheelCovTag);
    m.attr("WheelCovSlipTag") = py::str(WheelCovSlipTag);
    m.attr("WheelCovIntrTag") = py::str(WheelCovIntrTag);
    m.attr("WheelCovIntrSlipTag") = py::str(WheelCovIntrSlipTag);
    m.attr("WheelDangTag") = py::str(WheelDangTag);
    m.attr("PriorFactorIntrinsicsTag") = py::str(PriorFactorIntrinsicsTag);
    m.attr("PlanarPriorTag") = py::str(PlanarPriorTag);
    m.attr("ZPriorTag") = py::str(ZPriorTag);

    m.def("makeRobustParser", makeRobustParser);
    m.def("makeRobustWriter", makeRobustWriter);

    // ------------------------- Frontend ------------------------- //
    py::class_<JRLFrontend, boost::shared_ptr<JRLFrontend>>(m, "JRLFrontend")
        .def(py::init<boost::shared_ptr<FixedLagBackend>>(), "backend"_a)
        .def("run", &JRLFrontend::run);
    m.def(
        "makeFrontend",
        [](int kf = 5, int rf = 0, float spacing = 0) {
            boost::shared_ptr<FixedLagBackend> backend =
                boost::make_shared<MEstBackend>(gtsam::noiseModel::mEstimator::GemanMcClure::Create(3), "gm");
            ;
            backend->setKeyframeNum(kf);
            backend->setRegframeNum(rf);
            backend->setKeyframeSpace(spacing);
            return JRLFrontend(backend);
        },
        "kf"_a = 5, "rf"_a = 0, "spacing"_a = 0);

    // ------------------------- WheelFactors Bases ------------------------- //
    py::class_<PreintegratedWheelParams, boost::shared_ptr<PreintegratedWheelParams>>(m, "PreintegratedWheelParams")
        .def(py::init<>())
        .def_readwrite("wxCov", &PreintegratedWheelParams::wxCov)
        .def_readwrite("wyCov", &PreintegratedWheelParams::wyCov)
        .def_readwrite("vyCov", &PreintegratedWheelParams::vyCov)
        .def_readwrite("vzCov", &PreintegratedWheelParams::vzCov)
        .def_readwrite("omegaVelCov", &PreintegratedWheelParams::omegaVelCov)
        .def_readwrite("manCov", &PreintegratedWheelParams::manCov)
        .def_readwrite("manInitCov", &PreintegratedWheelParams::manInitCov)
        .def_readwrite("manPosCov", &PreintegratedWheelParams::manPosCov)
        .def_readwrite("manOrienCov", &PreintegratedWheelParams::manOrienCov)
        .def_readwrite("intrinsics", &PreintegratedWheelParams::intrinsics)
        .def_readwrite("intrinsicsBetweenCov", &PreintegratedWheelParams::intrinsicsBetweenCov)
        .def_static("MakeShared", &PreintegratedWheelParams::MakeShared)
        .def("makeFullCov", &PreintegratedWheelParams::makeFullCov)
        .def("makeFullVelCov", &PreintegratedWheelParams::makeFullVelCov)
        .def("setWVCovFromWheel", &PreintegratedWheelParams::setWVCovFromWheel)
        .def("setWVCovFromWV", &PreintegratedWheelParams::setWVCovFromWV)
        .def("intrinsicsMat", &PreintegratedWheelParams::intrinsicsMat);

    py::class_<PreintegratedWheelBase, boost::shared_ptr<PreintegratedWheelBase>>(m, "PreintegratedWheelBase")
        .def(py::init<boost::shared_ptr<PreintegratedWheelParams>>(), "params"_a)
        .def("preint", &PreintegratedWheelBase::preint)
        .def("params", &PreintegratedWheelBase::params)
        .def("deltaTij", &PreintegratedWheelBase::deltaTij)
        .def("resetIntegration", &PreintegratedWheelBase::resetIntegration)
        .def("preintMeasCov", &PreintegratedWheelBase::preintMeasCov)
        .def("integrateMeasurements", &PreintegratedWheelBase::integrateMeasurements);

    py::class_<WheelFactor2, gtsam::NoiseModelFactor, boost::shared_ptr<WheelFactor2>>(m, "WheelFactor2")
        .def(py::init<gtsam::Key, gtsam::Key, boost::shared_ptr<PreintegratedWheelBase>, gtsam::Pose3>(), "key1"_a,
             "key2"_a, "pwm"_a, "body_T_sensor"_a = gtsam::Pose3::Identity())
        .def("pwm", &WheelFactor2::pwm)
        .def("predict", &WheelFactor2::predict);

    py::class_<WheelFactor3, gtsam::NoiseModelFactor, boost::shared_ptr<WheelFactor3>>(m, "WheelFactor3")
        .def(py::init<gtsam::Key, gtsam::Key, gtsam::Key, boost::shared_ptr<PreintegratedWheelBase>, gtsam::Pose3>(),
             "key1"_a, "key2"_a, "key2"_a, "pwm"_a, "body_T_sensor"_a = gtsam::Pose3::Identity())
        .def("pwm", &WheelFactor3::pwm)
        .def("predict", &WheelFactor3::predict);

    py::class_<WheelFactor4Intrinsics, gtsam::NoiseModelFactor, boost::shared_ptr<WheelFactor4Intrinsics>>(
        m, "WheelFactor4Intrinsics")
        .def(py::init<gtsam::Key, gtsam::Key, gtsam::Key, gtsam::Key, boost::shared_ptr<PreintegratedWheelBase>,
                      gtsam::Pose3>(),
             "x_i"_a, "i_i"_a, "x_j"_a, "i_j"_a, "pwm"_a, "body_T_sensor"_a = gtsam::Pose3::Identity())
        .def("pwm", &WheelFactor4Intrinsics::pwm)
        .def("predict", &WheelFactor4Intrinsics::predict);

    py::class_<WheelFactor5, gtsam::NoiseModelFactor, boost::shared_ptr<WheelFactor5>>(m, "WheelFactor5")
        .def(py::init<gtsam::Key, gtsam::Key, gtsam::Key, gtsam::Key, gtsam::Key,
                      boost::shared_ptr<PreintegratedWheelBase>, gtsam::Pose3>(),
             "x_i"_a, "i_i"_a, "x_j"_a, "i_j"_a, "s"_a, "pwm"_a, "body_T_sensor"_a = gtsam::Pose3::Identity())
        .def("pwm", &WheelFactor5::pwm)
        .def("predict", &WheelFactor5::predict);

    py::class_<PlanarPriorFactor, gtsam::NoiseModelFactor, boost::shared_ptr<PlanarPriorFactor>>(m, "PlanarPriorFactor")
        .def(py::init<gtsam::Key, gtsam::Matrix2, gtsam::Pose3>(), "key1"_a, "cov"_a,
             "body_T_sensor"_a = gtsam::Pose3::Identity());

    py::class_<ZPriorFactor, gtsam::NoiseModelFactor, boost::shared_ptr<ZPriorFactor>>(m, "ZPriorFactor")
        .def(py::init<gtsam::Key, gtsam::Matrix1, gtsam::Pose3>(), "key1"_a, "cov"_a,
             "body_T_sensor"_a = gtsam::Pose3::Identity());

    // ------------------------- PreintegratedWheel Implementations ------------------------- //
    py::class_<PreintegratedWheelDang, boost::shared_ptr<PreintegratedWheelDang>, PreintegratedWheelBase>(
        m, "PreintegratedWheelDang")
        .def(py::init<boost::shared_ptr<PreintegratedWheelParams>>(), "params"_a)
        .def(py::init<PreintegratedWheelBase>(), "base"_a)
        .def("integrateVelocities", &PreintegratedWheelDang::integrateVelocities)
        .def(
            "predict", [](PreintegratedWheelDang *self, const gtsam::Pose3 &x1) { return self->predict(x1); }, "x1"_a)
        .def("copy", &PreintegratedWheelDang::copy);

    py::class_<PreintegratedWheelCov, boost::shared_ptr<PreintegratedWheelCov>, PreintegratedWheelBase>(
        m, "PreintegratedWheelCov")
        .def(py::init<boost::shared_ptr<PreintegratedWheelParams>>(), "params"_a)
        .def("integrateMeasurements", &PreintegratedWheelCov::integrateMeasurements)
        .def("preint_H_slip", &PreintegratedWheelCov::preint_H_slip)
        .def("preint_H_intr", &PreintegratedWheelCov::preint_H_intr)
        .def(
            "predict", [](PreintegratedWheelCov *self, const gtsam::Pose3 &x1) { return self->predict(x1); }, "x1"_a)
        .def("copy", &PreintegratedWheelCov::copy);
}