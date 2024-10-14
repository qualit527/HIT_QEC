// src/decoder_bindings.cpp

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "LLRBp4Decoder.h"

namespace py = pybind11;

PYBIND11_MODULE(LLRBP4_decoder, m) {
    m.doc() = "LLRBp4Decoder Python Bindings";

    // 定义 ScheduleType 枚举
    py::enum_<ScheduleType>(m, "ScheduleType")
        .value("FLOODING", ScheduleType::FLOODING)
        .value("LAYER", ScheduleType::LAYER)
        .export_values();

    // 定义 InitType 枚举
    py::enum_<InitType>(m, "InitType")
        .value("MOMENTUM", InitType::MOMENTUM)
        .value("NONE", InitType::NONE)
        .export_values();

    // 定义 MethodType 枚举
    py::enum_<MethodType>(m, "MethodType")
        .value("MOMENTUM", MethodType::MOMENTUM)
        .value("ADA", MethodType::ADA)
        .value("MBP", MethodType::MBP)
        .value("NONE", MethodType::NONE)
        .export_values();

    // 定义 OSDType 枚举
    py::enum_<OSDType>(m, "OSDType")
        .value("BINARY", OSDType::BINARY)
        .value("NONE", OSDType::NONE)
        .export_values();


    py::class_<LLRBp4Decoder>(m, "LLRBp4Decoder")
        .def(py::init<const Eigen::MatrixXi&,
                      const Eigen::MatrixXi&,
                      double,
                      double,
                      double,
                      int,
                      const Eigen::MatrixXi&,
                      double,
                      int>(),
             py::arg("Hx"),
             py::arg("Hz"),
             py::arg("px"),
             py::arg("py"),
             py::arg("pz"),
             py::arg("max_iter"),
             py::arg("Hs") = Eigen::MatrixXi(),
             py::arg("ps") = 1.0,
             py::arg("dimension") = 1)
        .def("standard_decoder",
             &LLRBp4Decoder::standard_decoder,
             py::arg("syndrome"),
             py::arg("schedule") = ScheduleType::FLOODING,
             py::arg("init") = InitType::NONE,
             py::arg("method") = MethodType::NONE,
             py::arg("OSD") = OSDType::NONE,
             py::arg("alpha") = 1.0,
             py::arg("beta") = 0.0);
}
