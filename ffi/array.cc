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
    Ref<Target> target;
    if (d.is_cpu()) {
        target = Ref<CPU>::make();
    } else if (d.is_cuda()) {
        target = Ref<GPU>::make();
    } else {
        throw DriverError("Unsupported PyTorch device");
    }
    if (d.has_index()) {
        return Ref<Device>::make(target, d.index());
    } else {
        return Ref<Device>::make(target);
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
    switch (dtype) {
    case DataType::Int32:
        return torch::ScalarType::Int;
    case DataType::Int64:
        return torch::ScalarType::Long;
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
    py::init([](py::array_t<nativeType, py::array::c_style> &np) {             \
        std::vector<size_t> shape(np.shape(), np.shape() + np.ndim());         \
        return Array::borrowFromRaw((void *)np.unchecked().data(), shape,      \
                                    dtype,                                     \
                                    Ref<Device>::make(Ref<CPU>::make()));      \
    }),                                                                        \
        "data"_a.noconvert(), py::keep_alive<1, 2>()

#define SHARE_TO_NUMPY(nativeType, dtype)                                      \
    case dtype: {                                                              \
        auto ptr = (const nativeType *)arr.rawSharedTo(                        \
            Ref<Device>::make(Ref<CPU>::make()));                              \
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
        .def(py::init([](const py::array &np) -> Ref<Array> {
            // Fallback holder. Don't let PyBind11 cast it automatically, or it
            // will all end up in float64 (the first initializer)
            throw DriverError(
                "Unsupported data type or strides from a NumPy Array. Please "
                "use freetensor.array factory function, instead of "
                "freetensor.Array, for strided arrays");
        }))
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

            for (size_t i = 0; i < lhs->size(); i++) {
                if (lptr[i] != rptr[i])
                    return false;
            }

            auto isDevIn = [&](const std::vector<ArrayCopy> &src,
                               const std::vector<ArrayCopy> &dst) {
                for (auto &&[ldev, lp, lb] : src) {
                    bool flg = 0;
                    for (auto &&[rdev, rp, rb] : dst) {
                        if (*ldev == *rdev) {
                            flg = 1;
                            break;
                        }
                    }
                    if (!flg)
                        return false;
                }
                return true;
            };
            if (!isDevIn(lhs->ptrs(), rhs->ptrs()) ||
                !isDevIn(rhs->ptrs(), lhs->ptrs()))
                return false;

            return true;
        });
#ifdef FT_WITH_PYTORCH
    pyArray.def(
        py::init([](const torch::Tensor &tensor) {
            if (tensor.is_contiguous()) {
                std::vector<size_t> shape(tensor.sizes().begin(),
                                          tensor.sizes().end());
                return Array::borrowFromRaw(
                    tensor.data_ptr(), shape,
                    dtypeFromPyTorch(tensor.scalar_type()),
                    deviceFromPyTorch(tensor.device()));
            } else {
                throw DriverError(
                    "Plese use freetensor.array factory function, instead of "
                    "freetensor.Array, for strided PyTorch tensors");
            }
        }),
        "data"_a.noconvert(), py::keep_alive<1, 2>());
#endif // FT_WITH_PYTORCH
    pyArray.def(
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
        py::keep_alive<0, 1>());
#ifdef FT_WITH_PYTORCH
    pyArray
        .def(
            "torch",
            [](Array &arr, const Ref<Device> &device) -> torch::Tensor {
                std::vector<int64_t> sizes(arr.shape().begin(),
                                           arr.shape().end());
                auto options = torch::TensorOptions(dtypeToPyTorch(arr.dtype()))
                                   .device(deviceToPyTorch(device));
                return torch::from_blob(arr.rawSharedTo(device), sizes,
                                        options);
            },
            py::keep_alive<0, 1>())
        .def(
            "torch",
            [](Array &arr) -> torch::Tensor {
                auto &&device = Config::defaultDevice();
                std::vector<int64_t> sizes(arr.shape().begin(),
                                           arr.shape().end());
                auto options = torch::TensorOptions(dtypeToPyTorch(arr.dtype()))
                                   .device(deviceToPyTorch(device));
                return torch::from_blob(arr.rawSharedTo(device), sizes,
                                        options);
            },
            py::keep_alive<0, 1>());
#endif // FT_WITH_PYTORCH
    pyArray.def_property_readonly("shape", &Array::shape)
        .def_property_readonly("dtype", &Array::dtype);
}

} // namespace freetensor
