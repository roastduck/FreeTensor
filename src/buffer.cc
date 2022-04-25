#include <buffer.h>
#include <hash.h>

namespace ir {

void Buffer::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace ir
