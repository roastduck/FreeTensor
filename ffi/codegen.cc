#include <codegen/code_gen.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_codegen(py::module_ &m) {
    m.def("code_gen", &codeGen, "func"_a, "target"_a);
    m.def("code_gen_cpu", &codeGenCPU, "func"_a);
    m.def("code_gen_cuda", &codeGenCUDA, "func"_a);
}

} // namespace freetensor
