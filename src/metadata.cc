#include <sstream>

#include <container_utils.h>
#include <hash.h>
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
    os << Indent(nIndent) << "$" << op_ << "{" << nl;
    for (auto &&[i, src] : views::enumerate(sources_)) {
        if (i != 0)
            os << ", " << nl;
        src->print(os, printLocation, nIndent + 1);
    }
    os << Indent(nIndent) << "}";
}

size_t TransformedMetadataContent::hash() const {
    size_t h = std::hash<int>{}((int)getType());
    h = hashCombine(h, std::hash<std::string>{}(op_));
    for (auto &&item : sources_) {
        h = hashCombine(h, item->hash());
    }
    return h;
}

bool TransformedMetadataContent::sameAs(const MetadataContent &_other) const {
    if (getType() != _other.getType()) {
        return false;
    }
    auto &other = (const TransformedMetadataContent &)_other;
    if (op_ != other.op_) {
        return false;
    }
    if (sources_.size() != other.sources_.size()) {
        return false;
    }
    for (auto &&[l, r] : views::zip(sources_, other.sources_)) {
        if (*l != *r) {
            return false;
        }
    }
    return true;
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
      location_(location), callerMetadata_(callerMetadata) {
    std::sort(labels_.begin(), labels_.end());
}

void SourceMetadataContent::print(std::ostream &os, bool printLocation,
                                  int nIndent) const {
    os << Indent(nIndent);
    for (auto &&[i, l] : views::enumerate(labels_)) {
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

size_t SourceMetadataContent::hash() const {
    size_t h = std::hash<int>{}((int)getType());
    for (auto &&item : labels_) {
        h = hashCombine(h, std::hash<std::string>{}(item));
    }
    if (location_.has_value()) {
        h = hashCombine(h, std::hash<std::string>{}(location_->first));
        h = hashCombine(h, std::hash<int>{}(location_->second));
    }
    if (callerMetadata_.isValid()) {
        h = hashCombine(h, callerMetadata_->hash());
    }
    return h;
}

bool SourceMetadataContent::sameAs(const MetadataContent &_other) const {
    if (getType() != _other.getType()) {
        return false;
    }
    auto &other = (const SourceMetadataContent &)_other;
    if (labels_ != other.labels_) {
        return false;
    }
    if (location_ != other.location_) {
        return false;
    }
    if (callerMetadata_.isValid() != other.callerMetadata_.isValid()) {
        return false;
    }
    if (callerMetadata_.isValid() && other.callerMetadata_.isValid() &&
        *callerMetadata_ != *other.callerMetadata_) {
        return false;
    }
    return true;
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
    if (os.iword(metadataPrintId) && id_.isValid()) {
        auto oldFlag = os.iword(OSTREAM_NO_ID_SIGN);
        os << manipNoIdSign(true) << Indent(nIndent) << "#" << id_;
        os.iword(OSTREAM_NO_ID_SIGN) = oldFlag;
    } else {
        os << Indent(nIndent) << "#<anon>";
    }
}

size_t AnonymousMetadataContent::hash() const {
    size_t h = std::hash<int>{}((int)getType());
    h = hashCombine(h, std::hash<ID>{}(id_));
    return h;
}

bool AnonymousMetadataContent::sameAs(const MetadataContent &_other) const {
    if (getType() != _other.getType()) {
        return false;
    }
    auto &other = (const AnonymousMetadataContent &)_other;
    if (id_ != other.id_) {
        return false;
    }
    return true;
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
