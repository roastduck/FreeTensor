#include <codegen/code_gen.h>
#include <codegen/code_gen_cpu.h>
#include <codegen/code_gen_cuda.h>

namespace freetensor {

std::string codeGen(const Func &func, const Ref<Target> &target) {
    switch (target->type()) {
    case TargetType::CPU:
        return codeGenCPU(func);
    case TargetType::GPU:
        return codeGenCUDA(func);
    default:
        ERROR("Unrecognized target " + target->toString());
    }
}

} // namespace freetensor
