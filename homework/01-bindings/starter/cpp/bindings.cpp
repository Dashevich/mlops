#include "kernel.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <array>
#include <string>

namespace py = pybind11;

namespace {

void validate_input(const py::array& value, const char* name) {
    if (!value.dtype().equal(py::dtype::of<double>())) {
        throw py::type_error(std::string("mac: '") + name +
                             "' must have native-endian float64 dtype, got " +
                             py::str(value.dtype()).cast<std::string>());
    }
    if (value.ndim() != 3) {
        throw py::value_error(std::string("mac: '") + name + "' must be 3D, got ndim=" +
                              std::to_string(value.ndim()));
    }
    if (!(value.flags() & py::array::c_style)) {
        throw py::value_error(std::string("mac: '") + name + "' must be C-contiguous");
    }
    if (!value.attr("flags").attr("aligned").cast<bool>()) {
        throw py::value_error(std::string("mac: '") + name + "' must be aligned");
    }
}

void require_same_shape(const py::array& reference, const py::array& value,
                        const char* name) {
    for (py::ssize_t axis = 0; axis < 3; ++axis) {
        if (value.shape(axis) != reference.shape(axis)) {
            throw py::value_error(std::string("mac: shape of '") + name +
                                  "' does not match shape of 'a' (broadcasting is not supported)");
        }
    }
}

}

py::array mac(const py::array& a, const py::array& b, const py::array& c) {
    validate_input(a, "a");
    validate_input(b, "b");
    validate_input(c, "c");
    require_same_shape(a, b, "b");
    require_same_shape(a, c, "c");

    const std::array<py::ssize_t, 3> shape{a.shape(0), a.shape(1), a.shape(2)};
    py::array_t<double, py::array::c_style> out(shape);

    const auto size = static_cast<std::size_t>(out.size());
    if (size == 0) {
        return out;
    }
    mac_kernel(static_cast<const double*>(a.data()),
               static_cast<const double*>(b.data()),
               static_cast<const double*>(c.data()),
               out.mutable_data(), size);
    return out;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Elementwise operation on three 3D float64 arrays";
    module.def("mac", &mac,
               "Return a * b + c elementwise for three C-contiguous, aligned, "
               "native-endian float64 arrays of identical 3D shape.",
               py::arg("a").noconvert(), py::arg("b").noconvert(), py::arg("c").noconvert());
}
