#ifndef DATA_TYPE_H
#define DATA_TYPE_H

#include <except.h>

namespace ir {

enum class DataType : int {
    Void = 0,
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Custom
};

inline std::string toString(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return "f32";
    case DataType::Float64:
        return "f64";
    case DataType::Int32:
        return "i32";
    case DataType::Int64:
        return "i64";
    case DataType::Bool:
        return "bool";
    case DataType::Custom:
        return "custom";
    case DataType::Void:
        return "void";
    }
    return "???";
}

inline size_t sizeOf(DataType dtype) {
    switch (dtype) {
    case DataType::Float64:
    case DataType::Int64:
        return 8;
    case DataType::Float32:
    case DataType::Int32:
        return 4;
    case DataType::Bool:
        return 1;
    case DataType::Custom:
        ERROR("Cannot get size of a customized data type");
    case DataType::Void:
        return 0;
    }
    ASSERT(false);
}

inline bool isInt(DataType dtype) {
    switch (dtype) {
    case DataType::Int32:
    case DataType::Int64:
        return true;
    default:
        return false;
    }
}

inline bool isFloat(DataType dtype) {
    switch (dtype) {
    case DataType::Float64:
    case DataType::Float32:
        return true;
    default:
        return false;
    }
}

inline bool isNumber(DataType dtype) { return isInt(dtype) || isFloat(dtype); }

inline bool isBool(DataType dtype) { return dtype == DataType::Bool; }

inline DataType upCast(DataType lhs, DataType rhs) {
    if (lhs == DataType::Custom || rhs == DataType::Custom) {
        return DataType::Custom;
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

} // namespace ir

#endif // DATA_TYPE
