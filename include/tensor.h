#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

namespace ir {

enum class DataType : int { Float32, Int32 };

inline std::string toString(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "f32";
    case DataType::Int32:
        return "i32";
    }
    return "???";
}

class Tensor {
    std::vector<int> shape_;
    DataType dtype_;

  public:
    Tensor(std::vector<int> &&shape, DataType dtype)
        : shape_(std::move(shape)), dtype_(dtype) {}
    Tensor(const std::vector<int> &shape, DataType dtype)
        : shape_(shape), dtype_(dtype) {}

    const std::vector<int> &shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
};

} // namespace ir

#endif // TENSOR_H
