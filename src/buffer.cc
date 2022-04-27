#include <buffer.h>
#include <hash.h>

namespace freetensor {

void Buffer::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace freetensor
