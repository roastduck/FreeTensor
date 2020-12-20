#include <ast.h>
#include <ffi.h>

namespace ir {

void init_ffi_ast(py::module_ &m) {
    m.def("makeVarDef", &makeVarDef);
    m.def("makeVar", &makeVar);
    m.def("makeStore", &makeStore);
    m.def("makeLoad", &makeLoad);
    m.def("makeIntConst", &makeIntConst);
    m.def("makeFloatConst", &makeFloatConst);
}

} // namespace ir

