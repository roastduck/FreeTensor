#include <ffi.h>
#include <selector.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_selector(py::module_ &m) {
    py::class_<Selector, Ref<Selector>>(m, "Selector")
        .def(
            py::init([](const std::string &str) { return parseSelector(str); }))
        .def("match", &Selector::match);
    py::implicitly_convertible<std::string, Selector>();
}

} // namespace freetensor
