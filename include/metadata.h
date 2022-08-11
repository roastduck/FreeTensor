#ifndef FREE_TENSOR_METADATA_H
#define FREE_TENSOR_METADATA_H

#include <atomic>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include <ref.h>

namespace freetensor {

class MetadataContent {
  protected:
    static void indent(std::ostream &os, int n) {
        for (int i = 0; i < n; ++i)
            os << "  ";
    }

  public:
    virtual ~MetadataContent() {}

    virtual void print(std::ostream &os, bool skipLocation,
                       int nIndent) const = 0;
};
using Metadata = Ref<MetadataContent>;

std::function<std::ostream &(std::ostream &)> skipLocation(bool skip);
std::ostream &operator<<(std::ostream &os, const Ref<MetadataContent> &mdc);

class TransformedMetadataContent : public MetadataContent {
    std::string op_;
    std::vector<Metadata> sources_;

  public:
    TransformedMetadataContent(const std::string &op,
                               const std::vector<Metadata> sources);

    ~TransformedMetadataContent() override = default;

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

    void print(std::ostream &os, bool skipLocation, int nIndent) const override;
};
using SourceMetadata = Ref<SourceMetadataContent>;

SourceMetadata
makeMetadata(const std::vector<std::string> &labels,
             const std::optional<std::pair<std::string, int>> &location,
             const Metadata &callerMetadata);

} // namespace freetensor

#endif // FREE_TENSOR_METADATA_H
