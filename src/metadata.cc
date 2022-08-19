#include <sstream>

#include <itertools.hpp>

#include <metadata.h>

namespace freetensor {

namespace {
const int metadataLocation = std::ios_base::xalloc();
const int metadataNewLine = std::ios_base::xalloc();
const int metadataPrintId = std::ios_base::xalloc();

struct Indent {
    int n;
};

std::ostream &operator<<(std::ostream &os, Indent indent) {
    if (os.iword(metadataNewLine))
        for (int i = 0; i < indent.n; ++i)
            os << "  ";
    return os;
}

std::ostream &nl(std::ostream &os) {
    if (os.iword(metadataNewLine))
        os.put('\n');
    return os;
}

} // namespace

std::ostream &manipMetadataSkipLocation(std::ostream &os) {
    os.iword(metadataLocation) = false;
    return os;
}
std::ostream &manipMetadataWithLocation(std::ostream &os) {
    os.iword(metadataLocation) = true;
    return os;
}

std::ostream &manipMetadataOneLine(std::ostream &os) {
    os.iword(metadataNewLine) = false;
    return os;
}
std::ostream &manipMetadataMultiLine(std::ostream &os) {
    os.iword(metadataNewLine) = true;
    return os;
}

std::ostream &manipMetadataNoId(std::ostream &os) {
    os.iword(metadataPrintId) = false;
    return os;
}
std::ostream &manipMetadataPrintId(std::ostream &os) {
    os.iword(metadataPrintId) = true;
    return os;
}

std::ostream &operator<<(std::ostream &os, const Ref<MetadataContent> &mdc) {
    if (mdc.isValid())
        mdc->print(os, os.iword(metadataLocation), 0);
    else
        makeMetadata()->print(os, os.iword(metadataLocation), 0);
    return os;
}

TransformedMetadataContent::TransformedMetadataContent(
    const std::string &op, const std::vector<Metadata> sources)
    : op_(op), sources_(sources) {
    for (const auto &source : sources_)
        ASSERT(source.isValid());
}

void TransformedMetadataContent::print(std::ostream &os, bool printLocation,
                                       int nIndent) const {
    os << Indent(nIndent) << "$" << op_ << " {" << nl;
    for (auto &&[i, src] : iter::enumerate(sources_)) {
        if (i != 0)
            os << ", " << nl;
        src->print(os, printLocation, nIndent + 1);
    }
    os << Indent(nIndent) << "}";
}

TransformedMetadata makeMetadata(const std::string &op,
                                 const std::vector<Metadata> &sources) {
    return Ref<TransformedMetadataContent>::make(op, sources);
}

SourceMetadataContent::SourceMetadataContent(
    const std::vector<std::string> &labels,
    const std::optional<std::pair<std::string, int>> &location,
    const Metadata &callerMetadata)
    : labels_(labels), labelsSet_(labels.begin(), labels.end()),
      location_(location), callerMetadata_(callerMetadata) {}

void SourceMetadataContent::print(std::ostream &os, bool printLocation,
                                  int nIndent) const {
    os << Indent(nIndent);
    for (auto &&[i, l] : iter::enumerate(labels_)) {
        if (i != 0)
            os << " ";
        os << l;
    }
    if (printLocation && location_)
        os << " @ " << location_->first << ":" << location_->second;
    if (callerMetadata_.isValid()) {
        os << " <~ " << nl;
        callerMetadata_->print(os, printLocation, nIndent);
    }
}

SourceMetadata
makeMetadata(const std::vector<std::string> &labels,
             const std::optional<std::pair<std::string, int>> &location,
             const Metadata &callerMetadata) {
    return Ref<SourceMetadataContent>::make(labels, location, callerMetadata);
}

AnonymousMetadataContent::AnonymousMetadataContent(const ID &id) : id_(id) {}

void AnonymousMetadataContent::print(std::ostream &os, bool skipLocation,
                                     int nIndent) const {
    if (os.iword(metadataPrintId) && id_.isValid())
        os << Indent(nIndent) << "#" << id_;
    else
        os << Indent(nIndent) << "#<anon>";
}

AnonymousMetadata makeMetadata(const ID &id) {
    return Ref<AnonymousMetadataContent>::make(id);
}

std::string toString(const Metadata &md, bool shouldSkipLocation) {
    std::ostringstream oss;
    if (shouldSkipLocation)
        oss << manipMetadataSkipLocation;
    oss << md;
    return oss.str();
}

} // namespace freetensor
