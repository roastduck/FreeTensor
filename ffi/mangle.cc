#include <ffi.h>
#include <serialize/mangle.h>

namespace freetensor {

void init_ffi_mangle(py::module_ &m) {
    m.def("mangle", &mangle);
    m.def("unmangle", &unmangle);
}

} // namespace freetensor
