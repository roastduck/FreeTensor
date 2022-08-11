#include <itertools.hpp>

#include <metadata.h>

namespace freetensor {

namespace {
const int metadataSkipLocation = std::ios_base::xalloc();
} // namespace

std::function<std::ostream &(std::ostream &)> skipLocation(bool skip) {
    return [skip](std::ostream &os) -> std::ostream & {
        os.iword(metadataSkipLocation) = skip;
        return os;
    };
}

std::ostream &operator<<(std::ostream &os, const Ref<MetadataContent> &mdc) {
    mdc->print(os, os.iword(metadataSkipLocation), 0);
    return os;
}

TransformedMetadataContent::TransformedMetadataContent(
    const std::string &op, const std::vector<Metadata> sources)
    : op_(op), sources_(sources) {}

void TransformedMetadataContent::print(std::ostream &os, bool skipLocation,
                                       int nIndent) const {
    indent(os, nIndent);
    os << op_ << " {\n";
    for (auto &&[i, src] : iter::enumerate(sources_)) {
        src->print(os, skipLocation, nIndent + 1);
        os << ",\n";
    }
    indent(os, nIndent);
    os << "}";
}

TransformedMetadata makeMetadata(const std::string &op,
                                 const std::vector<Metadata> &sources) {
    return Ref<TransformedMetadataContent>::make(op, sources);
}

SourceMetadataContent::SourceMetadataContent(
    const std::vector<std::string> &labels,
    const std::optional<std::pair<std::string, int>> &location,
    const Metadata &callerMetadata)
    : labels_(labels), location_(location), callerMetadata_(callerMetadata) {}

void SourceMetadataContent::print(std::ostream &os, bool skipLocation,
                                  int nIndent) const {
    indent(os, nIndent);
    for (auto &&[i, l] : iter::enumerate(labels_)) {
        if (i != 0)
            os << " ";
        os << l;
    }
    if (!skipLocation && location_)
        os << " @ " << location_->first << ":" << location_->second;
    if (callerMetadata_.isValid()) {
        os << " <-\n";
        callerMetadata_->print(os, skipLocation, nIndent);
    }
}

SourceMetadata
makeMetadata(const std::vector<std::string> &labels,
             const std::optional<std::pair<std::string, int>> &location,
             const Metadata &callerMetadata) {
    return Ref<SourceMetadataContent>::make(labels, location, callerMetadata);
}

} // namespace freetensor
