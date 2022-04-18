#ifndef ERROR_H_
#define ERROR_H_

#include <string>
#include <stdexcept>

#define ASSERT(expr) \
    if (!(expr)) { \
        throw std::runtime_error(__FILE__ ":" + std::to_string(__LINE__) +  ": Assertion " #expr " failed"); \
    }

#endif  // ERROR_H_
