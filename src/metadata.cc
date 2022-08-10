#include <itertools.hpp>

#include <metadata.h>

namespace freetensor {

std::ostream &operator<<(std::ostream &os, const Ref<MetadataContent> &mdc) {
    mdc->print(os);
    return os;
}

void TransformedMetadataContent::print(std::ostream &os, int nIndent) const {
    indent(os, nIndent);
    os << op_ << " {\n";
    for (auto &&[i, src] : iter::enumerate(sources_)) {
        src->print(os, nIndent + 1);
        os << ",\n";
    }
    indent(os, nIndent);
    os << "}";
}

void SourceMetadataContent::print(std::ostream &os, int nIndent) const {
    indent(os, nIndent);
    os << "[";
    for (auto &&[i, l] : iter::enumerate(labels_)) {
        if (i != 0)
            os << ", ";
        os << l;
    }
    os << "] @ ";
    for (auto &&[i, loc] : iter::enumerate(locationStack_)) {
        if (i != 0)
            os << " -> ";
        os << loc.first << ":" << loc.second;
    }
}

} // namespace freetensor
