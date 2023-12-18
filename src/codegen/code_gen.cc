#include <codegen/code_gen.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>

namespace freetensor {

NativeCode codeGen(const Func &func, const Ref<Target> &target) {
    switch (target->type()) {
    case TargetType::CPU:
        return codeGenCPU(func, target);
#ifdef FT_WITH_CUDA
    case TargetType::GPU:
        return codeGenCUDA(func, target);
#endif // FT_WITH_CUDA
    default:
        ERROR("Unrecognized target " + target->toString());
    }
}

} // namespace freetensor
