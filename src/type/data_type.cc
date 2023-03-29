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

SignDataType upCast(SignDataType lhs, SignDataType rhs) {
    if (isGT0(lhs) && isGT0(rhs)) {
        return SignDataType::GT0;
    } else if (isLT0(lhs) && isLT0(rhs)) {
        return SignDataType::LT0;
    } else if (isEQ0(lhs) && isEQ0(rhs)) {
        return SignDataType::EQ0;
    } else if (isGE0(lhs) && isGE0(rhs)) {
        return SignDataType::GE0;
    } else if (isLE0(lhs) && isLE0(rhs)) {
        return SignDataType::LE0;
    } else if (isNE0(lhs) && isNE0(rhs)) {
        return SignDataType::NE0;
    } else {
        return SignDataType::Any;
    }
}

} // namespace freetensor

namespace std {

size_t hash<freetensor::DataType>::operator()(
    const freetensor::DataType &dtype) const {
    return freetensor::hashCombine(h_((size_t)dtype.base()),
                                   h_((size_t)dtype.sign()));
}

} // namespace std

