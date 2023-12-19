#include <pybind11/numpy.h>
#include <vector>

#ifdef FT_WITH_PYTORCH
#include <torch/extension.h>
#include <torch/torch.h>
#endif

#include <config.h>
#include <debug.h>
#include <driver/array.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

#ifdef FT_WITH_PYTORCH
static Ref<Device> deviceFromPyTorch(const torch::Device &d) {
    TargetType targetType;
    if (d.is_cpu()) {
        targetType = TargetType::CPU;
    } else if (d.is_cuda()) {
#ifdef FT_WITH_CUDA
        targetType = TargetType::GPU;
#endif // FT_WITH_CUDA
    } else {
        throw DriverError("Unsupported PyTorch device");
    }
    if (d.has_index()) {
        return Ref<Device>::make(targetType, d.index());
    } else {
        return Ref<Device>::make(targetType);
    }
}

static torch::Device deviceToPyTorch(const Ref<Device> &d) {
    switch (d->type()) {
    case TargetType::CPU:
        // index = -1 : https://github.com/pytorch/pytorch/issues/79004
        return torch::Device(torch::DeviceType::CPU, -1);
    case TargetType::GPU:
        return torch::Device(torch::DeviceType::CUDA, d->num());
    default:
        throw DriverError("Unsupported device type by PyTorch");
    }
}

static DataType dtypeFromPyTorch(torch::ScalarType t) {
    switch (t) {
    case torch::ScalarType::Int:
        return DataType::Int32;
    case torch::ScalarType::Long:
        return DataType::Int64;
    case torch::ScalarType::Half:
        return DataType::Float16;
    case torch::ScalarType::Float:
        return DataType::Float32;
    case torch::ScalarType::Double:
        return DataType::Float64;
    case torch::ScalarType::Bool:
        return DataType::Bool;
    default:
        throw DriverError("Unsupported PyTorch data type");
    }
}

static torch::ScalarType dtypeToPyTorch(DataType dtype) {
    switch (dtype.base()) {
    case DataType::Int32:
        return torch::ScalarType::Int;
    case DataType::Int64:
        return torch::ScalarType::Long;
    case DataType::Float16:
        return torch::ScalarType::Half;
    case DataType::Float32:
        return torch::ScalarType::Float;
    case DataType::Float64:
        return torch::ScalarType::Double;
    case DataType::Bool:
        return torch::ScalarType::Bool;
    default:
        throw DriverError("Unsupported data type by PyTorch");
    }
}
#endif // FT_WITH_PYTORCH

void init_ffi_array(py::module_ &m) {
#define SHARE_FROM_NUMPY(nativeType, dtype)                                    \
    py::init([](py::array_t<nativeType, py::array::c_style> &np,               \
                bool dontDropBorrow, bool moved) {                             \
        std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());         \
        return Array::borrowFromRaw((void *)np.unchecked().data(), shape,      \
                                    dtype, Ref<Device>::make(TargetType::CPU), \
                                    dontDropBorrow, moved);                    \
    }),                                                                        \
        "data"_a.noconvert(),                                                  \
        "dont_drop_borrow"_a = false, "moved"_a = false,                       \
        py::keep_alive<1, 2>() /* Keep `np` alive whenever                     \
                                 `self` alives */

#define SHARE_TO_NUMPY(nativeType, dtype)                                      \
    case dtype: {                                                              \
        auto ptr = (const nativeType *)arr.rawSharedTo(                        \
            Ref<Device>::make(TargetType::CPU));                               \
        return py::array_t<nativeType>(arr.shape(), ptr,                       \
                                       py::capsule(ptr, [](void *) {}));       \
    }
    // Passing a capsule with an empty destructor to lend the data to NumPy
    // https://github.com/pybind/pybind11/issues/1042

    py::class_<Array, Ref<Array>> pyArray(m, "Array");
    pyArray.def(SHARE_FROM_NUMPY(double, DataType::Float64))
        .def(SHARE_FROM_NUMPY(float, DataType::Float32))
        .def(SHARE_FROM_NUMPY(int64_t, DataType::Int64))
        .def(SHARE_FROM_NUMPY(int32_t, DataType::Int32))
        .def(SHARE_FROM_NUMPY(bool, DataType::Bool))
        .def(py::init([](const py::array &np, bool, bool) -> Ref<Array> {
                 // Fallback holder. Don't let PyBind11 cast it automatically,
                 // or it will all end up in float64 (the first initializer)
                 throw DriverError(
                     "Unsupported data type or strides from a NumPy Array. "
                     "If you are using strided arrays, please use "
                     "freetensor.array factory function, instead of "
                     "freetensor.Array. If you are using float16, please use "
                     "the PyTorch interface instead.");
             }),
             "data"_a, "dont_drop_borrow"_a = false, "moved"_a = false)
        .def("__eq__", [](const Ref<Array> &lhs, const Ref<Array> &rhs) {
            /**
             * The feature is for testing serialization
             *
             * Note: lhs->ptrs_ / rhs->ptrs_ may be emplace_back(ed) by a
             * Arraycopy on CPU device
             */
            if (lhs->size() != rhs->size() || lhs->nElem() != rhs->nElem() ||
                lhs->dtype() != rhs->dtype() || lhs->shape() != rhs->shape())
                return false;

            uint8_t *lptr =
                (uint8_t *)lhs->rawSharedTo(Config::defaultDevice());
            uint8_t *rptr =
                (uint8_t *)rhs->rawSharedTo(Config::defaultDevice());

            if (memcmp(lptr, rptr, lhs->size()))
                return false;

            return true;
        });
#ifdef FT_WITH_PYTORCH
    pyArray.def(
        py::init([](const torch::Tensor &tensor, bool dontDropBorrow,
                    bool moved) {
            if (tensor.is_contiguous()) {
                std::vector<size_t> shape(tensor.sizes().begin(),
                                          tensor.sizes().end());
                return Array::borrowFromRaw(
                    tensor.data_ptr(), shape,
                    dtypeFromPyTorch(tensor.scalar_type()),
                    deviceFromPyTorch(tensor.device()), dontDropBorrow, moved);
            } else {
                throw DriverError(
                    "Plese use freetensor.array factory function, instead of "
                    "freetensor.Array, for strided PyTorch tensors");
            }
        }),
        "data"_a.noconvert(), "dont_drop_borrow"_a = false, "moved"_a = false,
        py::keep_alive<1,
                       2>() /* Keep `tensor` alive whenever `self` alives */);
#endif // FT_WITH_PYTORCH
    pyArray.def(
        "numpy",
        [](Array &arr) -> py::object {
            switch (arr.dtype().base()) {
                SHARE_TO_NUMPY(double, DataType::Float64)
                SHARE_TO_NUMPY(float, DataType::Float32)
                SHARE_TO_NUMPY(int64_t, DataType::Int64)
                SHARE_TO_NUMPY(int32_t, DataType::Int32)
                SHARE_TO_NUMPY(bool, DataType::Bool)
            case DataType::Float16:
                // TODO: Support fp16 after PyBind11 for NumPy fp16 is
                // available. Status:
                // https://github.com/pybind/pybind11/issues/1776,
                // https://github.com/pybind/pybind11/issues/4061
                throw DriverError(
                    "NumPy interface for float16 is not supported yet. Please "
                    "use the PyTorch interface instead.");
            default:
                ASSERT(false);
            }
        },
        py::keep_alive<
            0, 1>() /* Keep `self` alive whenever the return value alives */);
#ifdef FT_WITH_PYTORCH
    pyArray
        .def("torch",
             [](const Ref<Array> &arr,
                const Ref<Device> &device) -> torch::Tensor {
                 std::vector<int64_t> sizes(arr->shape().begin(),
                                            arr->shape().end());
                 auto options =
                     torch::TensorOptions(dtypeToPyTorch(arr->dtype()))
                         .device(deviceToPyTorch(device));
                 Ref<Array> refCntHolder = arr;
                 auto deleter = [refCntHolder = std::move(refCntHolder)](
                                    void *) mutable { refCntHolder = nullptr; };
                 return torch::from_blob(arr->rawSharedTo(device), sizes,
                                         deleter, options);
             })
        .def("torch", [](const Ref<Array> &arr) -> torch::Tensor {
            auto &&device = Config::defaultDevice();
            std::vector<int64_t> sizes(arr->shape().begin(),
                                       arr->shape().end());
            auto options = torch::TensorOptions(dtypeToPyTorch(arr->dtype()))
                               .device(deviceToPyTorch(device));
            Ref<Array> refCntHolder = arr;
            auto deleter = [refCntHolder = std::move(refCntHolder)](
                               void *) mutable { refCntHolder = nullptr; };
            return torch::from_blob(arr->rawSharedTo(device), sizes, deleter,
                                    options);
        });
#endif // FT_WITH_PYTORCH
    pyArray.def_property_readonly("shape", &Array::shape)
        .def_property_readonly("dtype", &Array::dtype)
        .def_property_readonly("dont_drop_borrow", &Array::dontDropBorrow)
        .def("set_dont_drop_borrow", &Array::setDontDropBorrow)
        .def_property_readonly("moved", &Array::moved)
        .def("set_moved", &Array::setMoved);
}

} // namespace freetensor
