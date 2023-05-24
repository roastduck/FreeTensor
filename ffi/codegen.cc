#include <codegen/code_gen.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>
#include <codegen/native_code.h>
#include <ffi.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_codegen(py::module_ &m) {
    py::class_<NativeCodeParam>(m, "NativeCodeParam")
        .def_readonly("name", &NativeCodeParam::name_)
        .def_readonly("dtype", &NativeCodeParam::dtype_)
        .def_readonly("atype", &NativeCodeParam::atype_)
        .def_readonly("mtype", &NativeCodeParam::mtype_)
        .def_readonly("update_closure", &NativeCodeParam::updateClosure_)
        .def_property_readonly("is_in_closure", &NativeCodeParam::isInClosure)
        .def("__str__",
             static_cast<std::string (*)(const NativeCodeParam &)>(&toString));

    py::class_<NativeCodeRet>(m, "NativCodeRet")
        .def_readonly("name", &NativeCodeRet::name_)
        .def_readonly("dtype", &NativeCodeRet::dtype_)
        .def_readonly("return_closure", &NativeCodeRet::returnClosure_)
        .def_property_readonly("is_in_closure", &NativeCodeRet::isInClosure)
        .def("__str__",
             static_cast<std::string (*)(const NativeCodeRet &)>(&toString));

    py::class_<NativeCode>(m, "NativeCode")
        .def(py::init<const std::string &, const std::vector<NativeCodeParam> &,
                      const std::vector<NativeCodeRet> &, const std::string &,
                      const Ref<Target> &>(),
             "name"_a, "params"_a, "returns"_a, "code"_a, "target"_a)
        .def(py::init(&NativeCode::fromFunc), "func"_a, "code"_a, "target"_a)
        .def_property_readonly("name", &NativeCode::name)
        .def_property_readonly("params", &NativeCode::params)
        .def_property_readonly("returns", &NativeCode::returns)
        .def_property_readonly("code", &NativeCode::code)
        .def_property_readonly("target", &NativeCode::target);

    m.def("code_gen", &codeGen, "func"_a, "target"_a);
    m.def("code_gen_cpu", &codeGenCPU, "func"_a);
    m.def("code_gen_cuda", &codeGenCUDA, "func"_a);
}

} // namespace freetensor
