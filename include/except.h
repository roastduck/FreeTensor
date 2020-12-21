#ifndef EXCEPT_H
#define EXCEPT_H

#include <stdexcept>
#include <string>

namespace ir {

#define ASSERT(expr)                                                           \
    {                                                                          \
        if (!(expr))                                                           \
            throw std::runtime_error("Assertion false: " #expr " at " __FILE__ \
                                     ":" +                                     \
                                     std::to_string(__LINE__));                \
    }

} // namespace ir

#endif // EXCEPT_H
