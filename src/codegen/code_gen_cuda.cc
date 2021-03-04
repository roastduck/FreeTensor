#include <analyze/normalize.h>
#include <codegen/code_gen_cuda.h>
#include <except.h>
#include <pass/simplify.h>

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

void CodeGenCUDA::visit(const ReduceTo &op) {
    if (op->atomic_) {
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
        CodeGenC::visit(op);
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
		if(op->unroll_num_ != 0) {
			os() << "#pragma unroll " << op->unroll_num_ << std::endl;
		}
        CodeGenC::visit(op);
    } else if (op->parallel_ == "blockIdx.x" || op->parallel_ == "blockIdx.y" ||
               op->parallel_ == "blockIdx.z" ||
               op->parallel_ == "threadIdx.x" ||
               op->parallel_ == "threadIdx.y" ||
               op->parallel_ == "threadIdx.z") {
        if (op->infoLen_->nodeType() != ASTNodeType::IntConst) {
            std::ostringstream msg;
            msg << "Length of " << op->parallel_
                << " should be constant, instead of " << op->infoLen_;
            throw Error(msg.str());
        }
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->parallel_] =
                op->infoLen_.as<IntConstNode>()->val_;
            endBlock();
            popStream();
            Stream &stream = poppedStream_.back();
            const auto &dim = stream.threadDim_;

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
                 << ")>>>(";
            bool first = true;
            for (auto &&item : stream.uses_) {
                os() << (first ? "" : ", ") << item.first;
                first = false;
            }
            os() << ");" << std::endl;
        } else {
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->parallel_] =
                op->infoLen_.as<IntConstNode>()->val_;
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

            makeIndent();
            beginBlock();

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
            endBlock();
            break;
        }

        case MemType::GPUShared: {
            markDef(normalizeId(op->name_), op->buffer_);

            makeIndent();
            beginBlock();

            makeIndent();

            // e.g. __shared__ float x[5][5][5];
            auto &&tensor = op->buffer_->tensor();
            auto &&shape = tensor.shape();
            os() << "__shared__ " << gen(tensor.dtype()) << " "
                 << normalizeId(op->name_);
            for (auto &&dim : shape) {
                if (dim->nodeType() != ASTNodeType::IntConst) {
                    throw Error("Shared memory buffer with dynamic size is not "
                                "supported yet");
                }
                os() << "[";
                (*this)(dim);
                os() << "]";
            }
            os() << ";" << std::endl;

            (*this)(op->body_);
            endBlock();
            break;
        }

        default:
            CodeGenC::visit(op);
            break;
        }
    }
}

std::pair<std::string, std::vector<std::string>> codeGenCUDA(const AST &_op) {
    auto op = simplifyPass(normalize(_op));

    CodeGenCUDA visitor;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header =
        "#include <cstdint>\n"
        "#include <algorithm>\n"
        "#define restrict __restrict__\n"
        "\n"
        "template <class T, size_t n> struct __ByValArray {\n"
        "    T data[n];\n"
        "    __host__ __device__ const T &operator[](size_t i) const {\n"
        "        return data[i];\n"
        "    }\n"
        "    __host__ __device__ T &operator[](size_t i) {\n"
        "        return data[i];\n"
        "    }\n"
        "};\n"
        "\n"
        "template <class T>\n"
        "T floorDiv(T a, T b) {\n"
        "  T res = a / b, rem = a % b;\n"
        "  return res - (rem != 0 && ((rem < 0) != (b < 0)));\n"
        "}\n"
        "template <class T>\n"
        "T ceilDiv(T a, T b) {\n"
        "  T res = a / b, rem = a % b;\n"
        "  return res + (rem != 0 && ((rem < 0) == (b < 0)));\n"
        "}\n"
        "\n"
        "extern \"C\" {\n"
        "\n";
    const char *tailer = "\n"
                         "}";

    auto body = visitor.toString([&](const CodeGenCUDA::Stream &stream) {
        if (stream.name_ == "default") {
            return "void run(void **_params) " + stream.os_.str();
        } else {
            std::ostringstream os;
            os << "__global__ void __launch_bounds__(";
            bool first = true;
            for (auto &&dim : stream.threadDim_) {
                os << (first ? "" : " * ") << dim.second;
                first = false;
            }
            os << ") " << stream.name_ << "(";
            first = true;
            for (auto &&item : stream.uses_) {
                os << (first ? "" : ", ");
                auto &&buffer = item.second;
                auto &&tensor = buffer->tensor();
                auto &&shape = tensor.shape();

                // FIXME: Normalize the ID (item.first)?
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
    return std::make_pair(header + body + tailer, visitor.params());
}

} // namespace ir

