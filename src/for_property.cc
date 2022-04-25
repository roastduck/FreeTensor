#include <for_property.h>
#include <hash.h>

namespace ir {

void ReductionItem::compHash() { hash_ = Hasher::compHash(*this); }

void ForProperty::compHash() { hash_ = Hasher::compHash(*this); }

} // namespace ir
