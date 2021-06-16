#include <codegen/code_gen_cuda.h>
#include <except.h>
#include <pass/simplify.h>

#include "detail/code_gen_c.h"

namespace ir {

bool CodeGenCUDA::inKernel() const {
    return streamStack_.back().name_ != "default";
}

void CodeGenCUDA::visit(const Min &op) {
    if (inKernel()) {
        os() << "min(";
        (*this)(op->lhs_);
        os() << ", ";
        (*this)(op->rhs_);
        os() << ")";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Max &op) {
    if (inKernel()) {
        os() << "max(";
        (*this)(op->lhs_);
        os() << ", ";
        (*this)(op->rhs_);
        os() << ")";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const Sqrt &op) {
    os() << "runtime_sqrt("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const Exp &op) {
    os() << "runtime_exp("; // Defined in runtime/gpu_runtime.h
    (*this)(op->expr_);
    os() << ")";
}

void CodeGenCUDA::visit(const ReduceTo &op) {
    auto id = normalizeId(op->var_);
    markUse(id);
    makeIndent();

    auto genAddr = [&]() {
        if (op->indices_.empty()) {
            os() << "*" << id;
        } else {
            os() << id;
            for (auto &&index : op->indices_) {
                os() << "[";
                (*this)(index);
                os() << "]";
            }
        }
    };
    auto genExpr = [&]() { (*this)(op->expr_); };

    if (op->atomic_) {
        switch (op->op_) {
        case ReduceOp::Add:
            os() << "atomicAdd(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Min:
            os() << "atomicMin(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        case ReduceOp::Max:
            os() << "atomicMax(&", genAddr(), os() << ", ", genExpr();
            os() << ");" << std::endl;
            break;
        default:
            ASSERT(false);
        }
    } else {
        switch (op->op_) {
        case ReduceOp::Add:
            genAddr(), os() << " += ", genExpr();
            break;
        case ReduceOp::Mul:
            genAddr(), os() << " *= ", genExpr();
            break;
        case ReduceOp::Min:
            genAddr(), os() << " = min(";
            genAddr(), os() << ", ", genExpr(), os() << ")";
            break;
        case ReduceOp::Max:
            genAddr(), os() << " = max(";
            genAddr(), os() << ", ", genExpr(), os() << ")";
            break;
        default:
            ASSERT(false);
        }
        os() << ";" << std::endl;
    }
}

void CodeGenCUDA::visit(const Var &op) {
    if (op->name_ == ".threadIdx.x") {
        os() << "(int)threadIdx.x";
    } else if (op->name_ == ".threadIdx.y") {
        os() << "(int)threadIdx.y";
    } else if (op->name_ == ".threadIdx.z") {
        os() << "(int)threadIdx.z";
    } else if (op->name_ == ".blockIdx.x") {
        os() << "(int)blockIdx.x";
    } else if (op->name_ == ".blockIdx.y") {
        os() << "(int)blockIdx.y";
    } else if (op->name_ == ".blockIdx.z") {
        os() << "(int)blockIdx.z";
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const For &op) {
    if (op->parallel_.empty()) {
        if (op->unroll_) {
            os() << "#pragma unroll " << op->len_ << std::endl;
        }
        CodeGenC::visit(op);
    } else if (op->parallel_ == "blockIdx.x" || op->parallel_ == "blockIdx.y" ||
               op->parallel_ == "blockIdx.z" ||
               op->parallel_ == "threadIdx.x" ||
               op->parallel_ == "threadIdx.y" ||
               op->parallel_ == "threadIdx.z") {
        if (op->len_->nodeType() != ASTNodeType::IntConst) {
            std::ostringstream msg;
            msg << "Length of " << op->parallel_
                << " should be constant, instead of " << op->len_;
            throw Error(msg.str());
        }
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->parallel_] =
                op->len_.as<IntConstNode>()->val_;
            endBlock();
            popStream();
            Stream &stream = poppedStream_.back();
            const auto &dim = stream.threadDim_;
            auto sharedSize = stream.sharedSize_;

            makeIndent();
            os() << kernel << "<<<dim3("
                 << (dim.count("blockIdx.x") ? dim.at("blockIdx.x") : 1) << ", "
                 << (dim.count("blockIdx.y") ? dim.at("blockIdx.y") : 1) << ", "
                 << (dim.count("blockIdx.z") ? dim.at("blockIdx.z") : 1)
                 << "), dim3("
                 << (dim.count("threadIdx.x") ? dim.at("threadIdx.x") : 1)
                 << ", "
                 << (dim.count("threadIdx.y") ? dim.at("threadIdx.y") : 1)
                 << ", "
                 << (dim.count("threadIdx.z") ? dim.at("threadIdx.z") : 1)
                 << "), " << std::to_string(sharedSize) << ">>>(";
            bool first = true;
            for (auto &&item : stream.uses_) {
                os() << (first ? "" : ", ") << item.first;
                first = false;
            }
            os() << ");" << std::endl;
        } else {
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->parallel_] =
                op->len_.as<IntConstNode>()->val_;
        }
    } else {
        throw Error("Unsupported parallel method" + op->parallel_);
    }
}

void CodeGenCUDA::visit(const VarDef &op) {
    if (op->buffer_->atype() != AccessType::Cache) {
        CodeGenC::visit(op);

    } else {
        switch (op->buffer_->mtype()) {
        case MemType::GPUGlobal: {
            if (inKernel()) {
                throw Error("Allocating a global buffer inside a kernel is not "
                            "supported yet");
            }

            markDef(normalizeId(op->name_), op->buffer_);

            // e.g.
            // float (*x)[5][5];  // CUDA does not allow "restrict" here
            // cudaMalloc(&x, 5 * 5 * 5 * sizeof(float)); ...; cudaFree(x);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor.shape();
            makeIndent();
            os() << gen(tensor.dtype()) << " (*";
            os() << normalizeId(op->name_) << ")";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << ";" << std::endl;
            makeIndent();
            os() << "cudaMalloc(&" << normalizeId(op->name_) << ", ";
            for (auto &&dim : shape) {
                (*this)(dim);
                os() << " * ";
            }
            os() << "sizeof(" << gen(tensor.dtype()) << "));" << std::endl;

            (*this)(op->body_);

            makeIndent();
            os() << "cudaFree(" << normalizeId(op->name_) << ");" << std::endl;
            break;
        }

        case MemType::GPUShared: {
            markDef(normalizeId(op->name_), op->buffer_);

            // A static shared memory array cannot be larger than 48KB (maybe a
            // bug of NVCC), so we allocate shared memory dynamically
            // e.g. float (*x)[5][5] = (float(*)[5][5])(__shmem + 0);
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor.shape();
            makeIndent();
            os() << gen(tensor.dtype()) << " (*";
            os() << normalizeId(op->name_) << ")";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << " = (" << gen(tensor.dtype()) << "(*)";
            for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                 i++) { // No shape[0]
                os() << "[";
                (*this)(shape[i]);
                os() << "]";
            }
            os() << ")(__shmem + " + std::to_string(sharedStackTop_) << ");"
                 << std::endl;

            int size = sizeOf(tensor.dtype());
            for (auto &&dim : shape) {
                if (dim->nodeType() == ASTNodeType::IntConst) {
                    size *= dim.as<IntConstNode>()->val_;
                } else {
                    throw InvalidProgram("Currently dynamic sized gpu/shared "
                                         "memory is not supported");
                }
            }

            streamStack_.back().sharedSize_ = std::max(
                streamStack_.back().sharedSize_, sharedStackTop_ + size);

            sharedStackTop_ += size;
            (*this)(op->body_);
            sharedStackTop_ -= size;

            break;
        }

        default:
            CodeGenC::visit(op);
            break;
        }
    }
}

std::string codeGenCUDA(const Func &func) {
    CodeGenCUDA visitor(func->params_);
    auto &&op = func->body_;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = R"~~~(
#include <gpu_runtime.h>

extern __shared__ uint8_t __shmem[];

extern "C" {
)~~~";
    const char *tailer = R"~~~(
}
)~~~";

    auto body = visitor.toString([&](const CodeGenCUDA::Stream &stream) {
        if (stream.name_ == "default") {
            return "void run(void **_params) " + stream.os_.str();
        } else {
            const auto &dim = stream.threadDim_;
            std::ostringstream os;
            os << "__global__ void __launch_bounds__(";
            os << (dim.count("threadIdx.x") ? dim.at("threadIdx.x") : 1);
            os << " * ";
            os << (dim.count("threadIdx.y") ? dim.at("threadIdx.y") : 1);
            os << " * ";
            os << (dim.count("threadIdx.z") ? dim.at("threadIdx.z") : 1);
            os << ") " << stream.name_ << "(";
            bool first = true;
            for (auto &&item : stream.uses_) {
                os << (first ? "" : ", ");
                auto &&buffer = item.second;
                auto &&tensor = buffer->tensor();
                auto &&shape = tensor.shape();

                switch (buffer->mtype()) {
                case MemType::ByValue:
                    // e.g.
                    // __ByValArray<__ByValArray<float, 2>, 2> x;
                    for (size_t i = 0, iEnd = shape.size(); i < iEnd; i++) {
                        os << "__ByValArray<";
                    }
                    os << CodeGenCUDA::gen(tensor.dtype());
                    for (auto it = shape.rbegin(); it != shape.rend(); it++) {
                        ASSERT((*it)->nodeType() == ASTNodeType::IntConst);
                        os << ", " << (*it).as<IntConstNode>()->val_ << ">";
                    }
                    os << " " << item.first;
                    break;

                default:
                    // e.g. const float (*restrict x)[5][5]
                    if (buffer->atype() == AccessType::Input) {
                        os << "const ";
                    }
                    os << CodeGenCUDA::gen(tensor.dtype()) << " (*restrict ";
                    os << item.first << ")";
                    for (size_t i = 1, iEnd = shape.size(); i < iEnd;
                         i++) { // No shape[0]
                        ASSERT(shape[i]->nodeType() == ASTNodeType::IntConst);
                        os << "[" << shape[i].as<IntConstNode>()->val_ << "]";
                    }
                }
                first = false;
            }
            os << ") ";
            os << stream.os_.str() << std::endl;
            return os.str();
        }
    });
    return header + body + tailer;
}

} // namespace ir
