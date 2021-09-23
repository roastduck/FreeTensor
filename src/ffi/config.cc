#include <string>

#include <ffi.h>

#define NAME_(macro) #macro
#define NAME(macro) NAME_(macro)

namespace ir {

using namespace pybind11::literals;

void init_ffi_config(py::module_ &m) {
    m.def("with_mkl", []() -> std::string {
#ifdef WITH_MKL
        return NAME(WITH_MKL);
#else
        return "";
#endif
    });
}

} // namespace ir
