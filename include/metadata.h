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
    virtual ~MetadataContent() = 0;

    virtual void print(std::ostream &os, int nIndent = 0) const = 0;
};
std::ostream &operator<<(std::ostream &os, const Ref<MetadataContent> &mdc);

using Metadata = Ref<MetadataContent>;

class TransformedMetadataContent : public MetadataContent {
    std::string op_;
    std::vector<Metadata> sources_;

  public:
    TransformedMetadataContent(const std::string &op,
                               const std::vector<Metadata> sources)
        : op_(op), sources_(sources) {}

    void print(std::ostream &os, int nIndent) const override;
};

Metadata makeMetadata(const std::string &op,
                      const std::vector<Metadata> &sources) {
    return Ref<TransformedMetadataContent>::make(op, sources);
}

class SourceMetadataContent : public MetadataContent {
    std::vector<std::string> labels_;
    std::vector<std::pair<std::string, int>> locationStack_;

  public:
    SourceMetadataContent(
        const std::vector<std::string> &labels,
        const std::vector<std::pair<std::string, int>> &locationStack)
        : labels_(labels), locationStack_(locationStack) {}

    void print(std::ostream &os, int nIndent) const override;
};

inline Metadata
makeMetadata(const std::vector<std::string> &labels,
             const std::vector<std::pair<std::string, int>> &locationStack) {
    return Ref<SourceMetadataContent>::make(labels, locationStack);
}

} // namespace freetensor

#endif // FREE_TENSOR_METADATA_H
