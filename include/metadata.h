#ifndef FREE_TENSOR_METADATA_H
#define FREE_TENSOR_METADATA_H

#include <atomic>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

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
    virtual void print(std::ostream &os, bool skipLocation,
                       int nIndent) const = 0;
};
using Metadata = Ref<MetadataContent>;

std::ostream &manipMetadataSkipLocation(std::ostream &);
std::ostream &manipMetadataWithLocation(std::ostream &);

std::ostream &manipMetadataOneLine(std::ostream &);
std::ostream &manipMetadataMultiLine(std::ostream &);

std::ostream &operator<<(std::ostream &os, const Metadata &md);

class TransformedMetadataContent : public MetadataContent {
    std::string op_;
    std::vector<Metadata> sources_;

  public:
    TransformedMetadataContent(const std::string &op,
                               const std::vector<Metadata> sources);

    ~TransformedMetadataContent() override = default;

    MetadataType getType() const override { return MetadataType::Transformed; }
    void print(std::ostream &os, bool skipLocation, int nIndent) const override;
};
using TransformedMetadata = Ref<TransformedMetadataContent>;

TransformedMetadata makeMetadata(const std::string &op,
                                 const std::vector<Metadata> &sources);

class SourceMetadataContent : public MetadataContent {
    std::vector<std::string> labels_;
    std::optional<std::pair<std::string, int>> location_;
    Metadata callerMetadata_;

  public:
    SourceMetadataContent(
        const std::vector<std::string> &labels,
        const std::optional<std::pair<std::string, int>> &location,
        const Metadata &callerMetadata);

    ~SourceMetadataContent() override = default;

    const std::vector<std::string> &labels() const { return labels_; }

    MetadataType getType() const override { return MetadataType::Source; }
    void print(std::ostream &os, bool skipLocation, int nIndent) const override;
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
    void print(std::ostream &os, bool skipLocation, int nIndent) const override;
};
using AnonymousMetadata = Ref<AnonymousMetadataContent>;

AnonymousMetadata makeMetadata(const ID &id = {});

std::string toString(const Metadata &md, bool shouldSkipLocation = false);

} // namespace freetensor

#endif // FREE_TENSOR_METADATA_H
