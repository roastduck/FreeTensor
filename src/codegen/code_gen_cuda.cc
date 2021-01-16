#include <codegen/code_gen_cuda.h>
#include <except.h>

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

void CodeGenCUDA::visit(const Var &op) {
    if (varMap_.count(op->name_)) {
        auto info = varMap_.at(op->name_);
        os() << info.threadId_ << " + ";
        (*this)(info.offset_);
    } else {
        CodeGenC::visit(op);
    }
}

void CodeGenCUDA::visit(const For &op) {
    if (op->parallel_.empty()) {
        CodeGenC::visit(op);
    } else if (op->parallel_ == "blockIdx.x" || op->parallel_ == "blockIdx.y" ||
               op->parallel_ == "blockIdx.z" ||
               op->parallel_ == "threadIdx.x" ||
               op->parallel_ == "threadIdx.y" ||
               op->parallel_ == "threadIdx.z") {
        if (op->info_len_->nodeType() != ASTNodeType::IntConst) {
            std::ostringstream msg;
            msg << "Length of " << op->parallel_
                << " should be constant, instead of " << op->info_len_;
            throw Error(msg.str());
        }
        varMap_[op->iter_] = {op->parallel_, op->begin_};
        if (!inKernel()) {
            std::string kernel = "kernel" + std::to_string(nKernel_++);
            pushStream(kernel);
            beginBlock();
            (*this)(op->body_);
            streamStack_.back().threadDim_[op->parallel_] =
                op->info_len_.as<IntConstNode>()->val_;
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
                op->info_len_.as<IntConstNode>()->val_;
        }
    } else {
        throw Error("Unsupported parallel method" + op->parallel_);
    }
}

std::pair<std::string, std::vector<std::string>> codeGenCUDA(const AST &op) {
    CodeGenCUDA visitor;
    visitor.beginBlock();
    visitor(op);
    visitor.endBlock();

    const char *header = "#include <cstdint>\n"
                         "#include <algorithm>\n"
                         "#define restrict __restrict__\n"
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

                // e.g. const float (*restrict x)[5][5]
                if (buffer->atype() == AccessType::Input) {
                    os << "const ";
                }
                os << CodeGenCUDA::gen(tensor.dtype()) << " (*restrict ";
                os << item.first << ")"; // FIXME: Normalize the ID?
                for (size_t i = 1, iEnd = tensor.shape().size(); i < iEnd;
                     i++) { // No shape[0]
                    ASSERT(tensor.shape()[i]->nodeType() ==
                           ASTNodeType::IntConst);
                    os << "[" << tensor.shape()[i].as<IntConstNode>()->val_
                       << "]";
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

