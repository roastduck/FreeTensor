#include <pybind11/numpy.h>
#include <vector>

#include <driver/array.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_array(py::module_ &m) {
#define SHARE_FROM_NUMPY(nativeType, dtype)                                    \
    py::init([](py::array_t<nativeType, py::array::c_style> &np) {             \
        std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());         \
        return Array::borrowFromRaw((void *)np.unchecked().data(), shape,      \
                                    dtype,                                     \
                                    Ref<Device>::make(Ref<CPU>::make()));      \
    }),                                                                        \
        py::keep_alive<1, 2>()

#define SHARE_TO_NUMPY(nativeType, dtype)                                      \
    case dtype: {                                                              \
        auto ptr = (const nativeType *)arr.rawSharedTo(                        \
            Ref<Device>::make(Ref<CPU>::make()));                              \
        return py::array_t<nativeType>(arr.shape(), ptr,                       \
                                       py::capsule(ptr, [](void *) {}));       \
    }
    // Passing a capsule with an empty destructor to lend the data to NumPy
    // https://github.com/pybind/pybind11/issues/1042

    py::class_<Array, Ref<Array>>(m, "Array")
        .def(SHARE_FROM_NUMPY(double, DataType::Float64))
        .def(SHARE_FROM_NUMPY(float, DataType::Float32))
        .def(SHARE_FROM_NUMPY(int64_t, DataType::Int64))
        .def(SHARE_FROM_NUMPY(int32_t, DataType::Int32))
        .def(SHARE_FROM_NUMPY(bool, DataType::Bool))
        .def(py::init([](const py::array &np) -> Ref<Array> {
            // Fallback holder. Don't let PyBind11 cast it automatically, or it
            // will all end up in float64 (the first initializer)
            throw DriverError(
                "Unsupported data type or strides from a NumPy Array");
        }))
        .def(
            "numpy",
            [](Array &arr) -> py::object {
                switch (arr.dtype()) {
                    SHARE_TO_NUMPY(double, DataType::Float64)
                    SHARE_TO_NUMPY(float, DataType::Float32)
                    SHARE_TO_NUMPY(int64_t, DataType::Int64)
                    SHARE_TO_NUMPY(int32_t, DataType::Int32)
                    SHARE_TO_NUMPY(bool, DataType::Bool)
                default:
                    ASSERT(false);
                }
            },
            py::keep_alive<0, 1>())
        .def_property_readonly("shape", &Array::shape)
        .def_property_readonly("dtype", &Array::dtype);
}

} // namespace freetensor
