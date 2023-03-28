#include <hash.h>
#include <type/data_type.h>

namespace freetensor {

size_t sizeOf(BaseDataType dtype) {
    switch (dtype) {
    case BaseDataType::Float64:
    case BaseDataType::Int64:
        return 8;
    case BaseDataType::Float32:
    case BaseDataType::Int32:
        return 4;
    case BaseDataType::Bool:
        return 1;
    case BaseDataType::Custom:
        ERROR("Cannot get size of a customized data type");
    case BaseDataType::Void:
        return 0;
    default:
        ASSERT(false);
    }
}

bool isInt(BaseDataType dtype) {
    switch (dtype) {
    case BaseDataType::Int32:
    case BaseDataType::Int64:
        return true;
    default:
        return false;
    }
}

bool isFloat(BaseDataType dtype) {
    switch (dtype) {
    case BaseDataType::Float64:
    case BaseDataType::Float32:
        return true;
    default:
        return false;
    }
}

BaseDataType upCast(BaseDataType lhs, BaseDataType rhs) {
    if (lhs == BaseDataType::Custom || rhs == BaseDataType::Custom) {
        return BaseDataType::Custom;
    }
    if (lhs == rhs) {
        return lhs;
    }
    if (isInt(lhs) && isFloat(rhs)) {
        return rhs;
    }
    if (isFloat(lhs) && isInt(rhs)) {
        return lhs;
    }
    if ((isInt(lhs) && isInt(rhs)) || (isFloat(lhs) && isFloat(rhs))) {
        return sizeOf(rhs) > sizeOf(lhs) ? rhs : lhs;
    }
    throw InvalidProgram("Cannot operate between " + toString(lhs) + " and " +
                         toString(rhs));
}

} // namespace freetensor

namespace std {

size_t hash<freetensor::DataType>::operator()(
    const freetensor::DataType &dtype) const {
    return freetensor::hashCombine(h_((size_t)dtype.base()),
                                   h_((size_t)dtype.sign()));
}

} // namespace std

