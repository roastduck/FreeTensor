#include <cursor.h>
#include <ffi.h>

namespace ir {

void init_ffi_cursor(py::module_ &m) {
    py::class_<Cursor>(m, "Cursor")
        .def("node", &Cursor::node)
        .def("nid", &Cursor::id)
        .def("node_type", &Cursor::nodeType)
        .def("prev", &Cursor::prev)
        .def("next", &Cursor::next)
        .def("outer", &Cursor::outer);
}

} // namespace ir

