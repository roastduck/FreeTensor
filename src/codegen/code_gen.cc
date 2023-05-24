#include <codegen/code_gen.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>

namespace freetensor {

NativeCode codeGen(const Func &func, const Ref<Target> &target) {
    switch (target->type()) {
    case TargetType::CPU:
        return NativeCode::fromFunc(func, codeGenCPU(func), target);
    case TargetType::GPU:
        return NativeCode::fromFunc(func, codeGenCUDA(func), target);
    default:
        ERROR("Unrecognized target " + target->toString());
    }
}

} // namespace freetensor
