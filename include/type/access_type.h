#ifndef FREE_TENSOR_ACCESS_TYPE_H
#define FREE_TENSOR_ACCESS_TYPE_H

#include <array>
#include <iostream>
#include <string>

#include <container_utils.h>
#include <serialize/to_string.h>

namespace freetensor {

/**
 * AccessType describes whether a tensor can be read or written
 *
 * An AccessType is decided by the following properties:
 *
 * - Writable: Writable in the program. The written result may or may not be
 * seen on user buffers.
 * - Inputting: The program needs user input to tensors of this type.
 * - Outputting: Users need output from tensors of this type. But even when a
 * tensor is not outputting, it may still be modified.
 *
 * Types for each combinations of the properties are:
 *
 * ```
 * Name             | Writable | Inputting | Outputting
 * NO USE           | NO       | NO        | NO
 * NO USE           | NO       | NO        | YES
 * input            | NO       | YES       | NO
 * bypass           | NO       | YES       | YES
 * cache            | YES      | NO        | NO
 * output           | YES      | NO        | YES
 * input-mutable(*) | YES      | YES       | NO
 * inout            | YES      | YES       | YES
 * ```
 *
 * (*) The written data may or may not be visible to user, depending on whether
 * the `Array` is `move`d.
 */
enum class AccessType : size_t {
    Input = 0,
    Bypass,
    Cache,
    Output,
    InputMutable,
    InOut,
    // ------
    NumTypes,
};

// First deduce array length, then assert, to ensure the length
constexpr std::array accessTypeNames = {
    "input", "bypass", "cache", "output", "input-mutable", "inout",
};
static_assert(accessTypeNames.size() == (size_t)AccessType::NumTypes);

inline std::ostream &operator<<(std::ostream &os, AccessType atype) {
    return os << accessTypeNames.at((size_t)atype);
}

inline AccessType parseAType(const std::string &_str) {
    auto &&str = tolower(_str);
    for (auto &&[i, s] : views::enumerate(accessTypeNames)) {
        if (s == str) {
            return (AccessType)i;
        }
    }
    std::string msg = "Unrecognized access type \"" + _str +
                      "\". Candidates are (case-insensitive): ";
    for (auto &&[i, s] : views::enumerate(accessTypeNames)) {
        msg += (i > 0 ? ", " : "");
        msg += s;
    }
    ERROR(msg);
}

inline bool isWritable(AccessType atype) {
    switch (atype) {
    case AccessType::Cache:
    case AccessType::Output:
    case AccessType::InputMutable:
    case AccessType::InOut:
        return true;
    default:
        return false;
    }
}

inline bool isInputting(AccessType atype) {
    switch (atype) {
    case AccessType::Input:
    case AccessType::Bypass:
    case AccessType::InputMutable:
    case AccessType::InOut:
        return true;
    default:
        return false;
    }
}

inline bool isOutputting(AccessType atype) {
    switch (atype) {
    case AccessType::Bypass:
    case AccessType::Output:
    case AccessType::InOut:
        return true;
    default:
        return false;
    }
}

inline AccessType addOutputting(AccessType atype) {
    switch (atype) {
    case AccessType::Input:
        return AccessType::Bypass;
    case AccessType::Cache:
        return AccessType::Output;
    case AccessType::InputMutable:
        return AccessType::InOut;
    default:
        return atype;
    }
}

inline AccessType removeOutputting(AccessType atype) {
    switch (atype) {
    case AccessType::Bypass:
        return AccessType::Input;
    case AccessType::Output:
        return AccessType::Cache;
    case AccessType::InOut:
        return AccessType::InputMutable;
    default:
        return atype;
    }
}

} // namespace freetensor

#endif // FREE_TENSOR_ACCESS_TYPE_H
