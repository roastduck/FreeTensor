#ifndef EXCEPT_H
#define EXCEPT_H

#include <iostream>
#include <stdexcept>
#include <string>

namespace ir {

#define ERROR(msg)                                                             \
    {                                                                          \
        throw std::runtime_error((std::string) "[ERROR] " __FILE__ ":" +       \
                                 std::to_string(__LINE__) + ": " + (msg));     \
    }

#define WARNING(msg)                                                           \
    {                                                                          \
        std::cerr << ((std::string) "[WARING] " __FILE__ ":" +                 \
                      std::to_string(__LINE__) + ": " + (msg));                \
    }

#define ASSERT(expr)                                                           \
    {                                                                          \
        if (!(expr)) {                                                         \
            ERROR("Assertion false: " #expr)                                   \
        }                                                                      \
    }

} // namespace ir

#endif // EXCEPT_H
