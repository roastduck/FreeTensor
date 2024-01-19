#ifndef FREE_TENSOR_DATA_TYPE_H
#define FREE_TENSOR_DATA_TYPE_H

#include <array>
#include <functional>

#include <container_utils.h>
#include <except.h>
#include <serialize/to_string.h>

namespace freetensor {

enum class BaseDataType : size_t {
    Void = 0, // Returns nothing. It is a Unit Type
    Float16,
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
    Custom,
    Never, // Never returns. It is the Bottom Type
    // ------
    NumTypes,
};

constexpr std::array baseDataTypeNames = {
    "void",  "float16", "float32", "float64", "int32",
    "int64", "bool",    "custom",  "never",
};
static_assert(baseDataTypeNames.size() == (size_t)BaseDataType::NumTypes);

namespace detail {

template <typename T, T... i>
constexpr auto createAllBaseDataTypes(std::integer_sequence<T, i...>) {
    return std::array{(BaseDataType)i...};
}

} // namespace detail

constexpr auto allBaseDataTypes = detail::createAllBaseDataTypes(
    std::make_index_sequence<(size_t)BaseDataType::NumTypes>{});

inline std::ostream &operator<<(std::ostream &os, BaseDataType dtype) {
    return os << baseDataTypeNames.at((size_t)dtype);
}

inline BaseDataType parseBaseDataType(const std::string &_str) {
    auto &&str = tolower(_str);
    for (auto &&[i, s] : views::enumerate(baseDataTypeNames)) {
        if (s == str) {
            return (BaseDataType)i;
        }
    }
    ERROR(FT_MSG << "Unrecognized base data type \"" << _str
                 << "\". Candidates are (case-insensitive): "
                 << (baseDataTypeNames | join(", ")));
}

enum class SignDataType : size_t {
    Any = 0,
    GT0,
    GE0,
    LT0,
    LE0,
    NE0,
    EQ0, // EQ0 is only for "0" literals. No need to type-inference EQ0 because
         // we have const_fold
    Never, // Bottom type
    // ------
    NumTypes,
};

constexpr std::array signDataTypeNames = {
    "", ">0", ">=0", "<0", "<=0", "!=0", "==0", "{}",
};
static_assert(signDataTypeNames.size() == (size_t)SignDataType::NumTypes);

namespace detail {

template <typename T, T... i>
constexpr auto createAllSignDataTypes(std::integer_sequence<T, i...>) {
    return std::array{(SignDataType)i...};
}

} // namespace detail

constexpr auto allSignDataTypes = detail::createAllSignDataTypes(
    std::make_index_sequence<(size_t)SignDataType::NumTypes>{});

inline std::ostream &operator<<(std::ostream &os, SignDataType dtype) {
    return os << signDataTypeNames.at((size_t)dtype);
}

inline SignDataType parseSignDataType(const std::string &str) {
    for (auto &&[i, s] : views::enumerate(signDataTypeNames)) {
        if (s == str) {
            return (SignDataType)i;
        }
    }
    ERROR(FT_MSG << "Unrecognized sign data type \"" << str
                 << "\". Candidates are: " << (signDataTypeNames | join(", ")));
}

class DataType {
    BaseDataType base_;
    SignDataType sign_;

  public:
    DataType() {} // Construct without initialization
    DataType(BaseDataType base, SignDataType sign = SignDataType::Any)
        : base_(base), sign_(sign) {}

    // Expose BaseDataType::* to DataType::*
    //
    // TODO: Use the following line after GCC 12.3. GCC is buggy with `using
    // enum` before 12.3 (https://gcc.gnu.org/bugzilla/show_bug.cgi?id=103081)
    //
    // using enum BaseDataType;
    //
    // and remove the following lines
    constexpr static auto Bool = BaseDataType::Bool;
    constexpr static auto Custom = BaseDataType::Custom;
    constexpr static auto Float16 = BaseDataType::Float16;
    constexpr static auto Float32 = BaseDataType::Float32;
    constexpr static auto Float64 = BaseDataType::Float64;
    constexpr static auto Int32 = BaseDataType::Int32;
    constexpr static auto Int64 = BaseDataType::Int64;
    constexpr static auto Void = BaseDataType::Void;

    const auto &base() const { return base_; }
    const auto &sign() const { return sign_; }

    friend bool operator==(const DataType &, const DataType &) = default;
};

inline std::ostream &operator<<(std::ostream &os, const DataType &dtype) {
    return os << dtype.base() << dtype.sign();
}

inline DataType parseDType(const std::string &str) {
    auto split = str.find_first_of("<>=!");
    if (split == std::string::npos) {
        split = str.length();
    }
    auto base = parseBaseDataType(str.substr(0, split));
    auto sign = parseSignDataType(str.substr(split));
    return DataType{base, sign};
}

size_t sizeOf(BaseDataType dtype);
inline size_t sizeOf(const DataType &dtype) { return sizeOf(dtype.base()); }

// The following functions tests properties of a type. NOTE: All properties hold
// for the bottom type `Never`, because $\forall x \in \emptyset : P(x)$ is
// always true for any $P$

bool isInt(BaseDataType dtype);
inline bool isInt(const DataType &dtype) { return isInt(dtype.base()); }

bool isFloat(BaseDataType dtype);
inline bool isFloat(const DataType &dtype) { return isFloat(dtype.base()); }

inline bool isNumber(BaseDataType dtype) {
    return isInt(dtype) || isFloat(dtype);
}
inline bool isNumber(const DataType &dtype) { return isNumber(dtype.base()); }

bool isBool(BaseDataType dtype);
inline bool isBool(const DataType &dtype) { return isBool(dtype.base()); }

bool isGT0(SignDataType dtype);
inline bool isGT0(const DataType &dtype) { return isGT0(dtype.sign()); }

bool isGE0(SignDataType dtype);
inline bool isGE0(const DataType &dtype) { return isGE0(dtype.sign()); }

bool isLT0(SignDataType dtype);
inline bool isLT0(const DataType &dtype) { return isLT0(dtype.sign()); }

bool isLE0(SignDataType dtype);
inline bool isLE0(const DataType &dtype) { return isLE0(dtype.sign()); }

bool isNE0(SignDataType dtype);
inline bool isNE0(const DataType &dtype) { return isNE0(dtype.sign()); }

bool isEQ0(SignDataType dtype);
inline bool isEQ0(const DataType &dtype) { return isNE0(dtype.sign()); }

/**
 * Union type
 *
 * Obtain a new data type containing as few as possible values, where the value
 * is of either of the input types. Please note that although union type on
 * BaseDataType can be used to predict the resulting type of some binary
 * operations, this is not true for all cases and does not apply to
 * SignDataType.
 *
 * @{
 */
BaseDataType upCast(BaseDataType lhs, BaseDataType rhs);
SignDataType upCast(SignDataType lhs, SignDataType rhs);
inline DataType upCast(const DataType &lhs, const DataType &rhs) {
    return {upCast(lhs.base(), rhs.base()), upCast(lhs.sign(), rhs.sign())};
}
/** @} */

/**
 * Intersect type
 *
 * Obatin a new data type containing as few as possible values, where the value
 * is of both of the input types. This is actually merging restrictions from
 * both types.
 */
BaseDataType downCast(BaseDataType lhs, BaseDataType rhs);
SignDataType downCast(SignDataType lhs, SignDataType rhs);
inline DataType downCast(const DataType &lhs, const DataType &rhs) {
    return {downCast(lhs.base(), rhs.base()), downCast(lhs.sign(), rhs.sign())};
}
/** @} */

} // namespace freetensor

namespace std {

template <> class hash<freetensor::DataType> {
    std::hash<size_t> h_;

  public:
    size_t operator()(const freetensor::DataType &dtype) const;
};

} // namespace std

#endif // FREE_TENSOR_DATA_TYPE
