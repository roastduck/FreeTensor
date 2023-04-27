#ifndef FREE_TENSOR_METADATA_H
#define FREE_TENSOR_METADATA_H

#include <atomic>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include <func_utils.h>
#include <id.h>
#include <ref.h>

namespace freetensor {

enum class MetadataType {
    Transformed,
    Source,
    Anonymous,
};

class MetadataContent {
  public:
    virtual ~MetadataContent() {}

    virtual MetadataType getType() const = 0;
    virtual bool printByDefault() const = 0;
    virtual void print(std::ostream &os, bool printLocation,
                       int nIndent) const = 0;

    virtual size_t hash() const = 0;
    virtual bool sameAs(const MetadataContent &other) const = 0;
    friend bool operator==(const MetadataContent &lhs,
                           const MetadataContent &rhs) {
        return lhs.sameAs(rhs);
    }
};
using Metadata = Ref<MetadataContent>;
using MetadataHasher = PtrInvocable<std::hash<MetadataContent>>;
using MetadataComparator = PtrInvocable<std::equal_to<MetadataContent>>;

std::ostream &manipMetadataSkipLocation(std::ostream &);
std::ostream &manipMetadataWithLocation(std::ostream &);

std::ostream &manipMetadataOneLine(std::ostream &);
std::ostream &manipMetadataMultiLine(std::ostream &);

std::ostream &manipMetadataNoId(std::ostream &os);
std::ostream &manipMetadataPrintId(std::ostream &os);

std::ostream &operator<<(std::ostream &os, const Metadata &md);

class TransformedMetadataContent : public MetadataContent {
    std::string op_;
    std::vector<Metadata> sources_;

  public:
    TransformedMetadataContent(const std::string &op,
                               const std::vector<Metadata> sources);

    ~TransformedMetadataContent() override = default;

    MetadataType getType() const override { return MetadataType::Transformed; }
    bool printByDefault() const override { return true; }
    void print(std::ostream &os, bool printLocation,
               int nIndent) const override;

    size_t hash() const override;
    bool sameAs(const MetadataContent &other) const override;

    const std::string &op() const { return op_; }
    const std::vector<Metadata> &sources() const { return sources_; }
};
using TransformedMetadata = Ref<TransformedMetadataContent>;

TransformedMetadata makeMetadata(const std::string &op,
                                 const std::vector<Metadata> &sources);

class SourceMetadataContent : public MetadataContent {
    std::vector<std::string> labels_; // sorted
    std::unordered_set<std::string> labelsSet_;
    std::optional<std::pair<std::string, int>> location_;
    Metadata callerMetadata_;

  public:
    SourceMetadataContent(
        const std::vector<std::string> &labels,
        const std::optional<std::pair<std::string, int>> &location,
        const Metadata &callerMetadata);

    ~SourceMetadataContent() override = default;

    const std::vector<std::string> &labels() const { return labels_; }
    const std::unordered_set<std::string> &labelsSet() const {
        return labelsSet_;
    }
    const Metadata &caller() const { return callerMetadata_; }

    MetadataType getType() const override { return MetadataType::Source; }
    bool printByDefault() const override { return !labels_.empty(); }
    void print(std::ostream &os, bool printLocation,
               int nIndent) const override;

    size_t hash() const override;
    bool sameAs(const MetadataContent &other) const override;
};
using SourceMetadata = Ref<SourceMetadataContent>;

SourceMetadata
makeMetadata(const std::vector<std::string> &labels,
             const std::optional<std::pair<std::string, int>> &location,
             const Metadata &callerMetadata);

class AnonymousMetadataContent : public MetadataContent {
    ID id_;

  public:
    AnonymousMetadataContent(const ID &id);
    ~AnonymousMetadataContent() override = default;
    MetadataType getType() const override { return MetadataType::Anonymous; }
    bool printByDefault() const override { return false; }
    void print(std::ostream &os, bool printLocation,
               int nIndent) const override;

    size_t hash() const override;
    bool sameAs(const MetadataContent &other) const override;

    ID id() const { return id_; }
};
using AnonymousMetadata = Ref<AnonymousMetadataContent>;

AnonymousMetadata makeMetadata(const ID &id = {});

std::string toString(const Metadata &md, bool shouldSkipLocation = false);

} // namespace freetensor

namespace std {

template <> struct hash<freetensor::MetadataContent> {
    size_t operator()(const freetensor::MetadataContent &md) const {
        return md.hash();
    }
};

} // namespace std

#endif // FREE_TENSOR_METADATA_H
