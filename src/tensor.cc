#include <tensor.h>

namespace ir {

static std::vector<Expr> vals2exprs(const std::vector<int> &vals) {
    std::vector<Expr> exprs;
    exprs.reserve(vals.size());
    for (auto &&val : vals) {
        exprs.emplace_back(makeIntConst(val));
    }
    return exprs;
}

static std::vector<Expr> vals2exprs(const std::vector<double> &vals) {
    std::vector<Expr> exprs;
    exprs.reserve(vals.size());
    for (auto &&val : vals) {
        exprs.emplace_back(makeFloatConst(val));
    }
    return exprs;
}

TensorData::TensorData(DataType dtype, const std::vector<int> &shape,
                       const std::vector<Expr> &data)
    : dtype_(dtype), shape_(shape), data_(data) {
    size_ = 1;
    for (auto dim : shape) {
        size_ *= dim;
    }
}

TensorData::TensorData(DataType dtype, const std::vector<int> &shape,
                       std::vector<Expr> &&data)
    : dtype_(dtype), shape_(shape), data_(std::move(data)) {
    size_ = 1;
    for (auto dim : shape) {
        size_ *= dim;
    }
}

TensorData::TensorData(const std::vector<int> &shape,
                       const std::vector<int> &data)
    : TensorData(DataType::Int32, shape, vals2exprs(data)) {}

TensorData::TensorData(const std::vector<int> &shape,
                       const std::vector<double> &data)
    : TensorData(DataType::Float32, shape, vals2exprs(data)) {}

std::vector<int> TensorData::indices(int offset) const {
    std::vector<int> ret(shape_.size(), 0);
    for (size_t i = shape_.size() - 1; ~i; i--) {
        ret[i] = offset % shape_[i];
        offset /= shape_[i];
    }
    return ret;
}

const Expr &TensorData::at(int offset) const { return data_.at(offset); }

} // namespace ir

