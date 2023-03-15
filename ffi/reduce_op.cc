#include <ffi.h>
#include <reduce_op.h>

namespace freetensor {

using namespace pybind11::literals;

void init_ffi_reduce_op(py::module_ &m) {
    py::enum_<ReduceOp>(m, "ReduceOp")
        .value("Add", ReduceOp::Add)
        .value("Max", ReduceOp::Max)
        .value("Min", ReduceOp::Min)
        .value("Mul", ReduceOp::Mul)
        .value("And", ReduceOp::LAnd)
        .value("Or", ReduceOp::LOr);

    m.def("neutral_val", &neutralVal);
}

} // namespace freetensor
