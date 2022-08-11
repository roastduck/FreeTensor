#include <sstream>

#include <ffi.h>
#include <metadata.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_metadata(py::module_ &m) {
    py::class_<MetadataContent, Metadata> pyMetadata(m, "Metadata");
    pyMetadata.def("__str__", [](const Metadata &metadata) {
        std::ostringstream oss;
        oss << metadata;
        return oss.str();
    });
    py::class_<SourceMetadataContent, Ref<SourceMetadataContent>>(
        m, "SourceMetadata", pyMetadata)
        .def(py::init(static_cast<Ref<SourceMetadataContent> (*)(
                          const std::vector<std::string> &,
                          const std::optional<std::pair<std::string, int>> &,
                          const Metadata &)>(&makeMetadata)),
             "labels"_a, "location"_a = std::nullopt,
             "callerMetadata"_a = nullptr);
}

} // namespace freetensor