#ifndef FREE_TENSOR_LOAD_TARGET_H
#define FREE_TENSOR_LOAD_TARGET_H
// for multi-machine-parallel xmlrpc


#include <ref.h>
#include <driver/target.h>

namespace freetensor{

Ref<Target> loadTarget(const std::string &txt);



} // namespace freetensor

#endif // FREE_TENSOR_LOAD_TARGET_H
