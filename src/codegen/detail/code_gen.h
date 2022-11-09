#ifndef DETAIL_CODE_GEN_H
#define DETAIL_CODE_GEN_H

#include <codegen/code_gen.h>
#include <serialize/print_ast.h>

namespace freetensor {

inline CodeGenStream::CodeGenStream() { os_.iword(OSTREAM_NO_PRETTY) = true; }

template <class Stream>
CodeGen<Stream>::CodeGen(bool compact, int indentSize)
    : compact_(compact), indentSize_(compact ? 0 : indentSize) {
    pushStream("default");
}

template <class Stream> void CodeGen<Stream>::beginBlock() {
    os() << "{" << std::endl;
    nIndent()++;
}

template <class Stream> void CodeGen<Stream>::endBlock() {
    nIndent()--;
    makeIndent();
    os() << "}" << std::endl;
}

template <class Stream> void CodeGen<Stream>::makeIndent() {
    for (int i = 0, iEnd = nIndent() * indentSize_; i < iEnd; i++) {
        os() << " ";
    }
}

template <class Stream> void CodeGen<Stream>::markDef(const VarDef &op) {
    var2Stream_[op->name_] = streamStack_.back().name_;
    pushDef(op);
}

template <class Stream> void CodeGen<Stream>::markUse(const std::string &name) {
    auto &&stream = var2Stream_.at(name);
    auto &&d = def(name);
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        if (it->name_ == stream) {
            break;
        }
        it->useDefs_[name] = d;
    }
}

template <class Stream> void CodeGen<Stream>::markUndef(const VarDef &op) {
    var2Stream_.erase(op->name_);
    popDef(op);
}

template <class Stream> void CodeGen<Stream>::markDefIter(const For &op) {
    var2Stream_[op->iter_] = streamStack_.back().name_;
    pushFor(op);
}

template <class Stream>
void CodeGen<Stream>::markUseIter(const std::string &name) {
    auto &&stream = var2Stream_.at(name);
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        if (it->name_ == stream) {
            break;
        }
        it->useIters_.insert(name);
    }
}

template <class Stream> void CodeGen<Stream>::markUndefIter(const For &op) {
    var2Stream_.erase(op->iter_);
    popFor(op);
}

template <class Stream>
void CodeGen<Stream>::pushStream(const std::string &name) {
    streamStack_.emplace_back();
    streamStack_.back().name_ = name;
}

template <class Stream> void CodeGen<Stream>::popStream() {
    poppedStream_.emplace_back(std::move(streamStack_.back()));
    streamStack_.pop_back();
}

template <class Stream> std::ostream &CodeGen<Stream>::os() {
    return streamStack_.back().os_;
}

template <class Stream> int &CodeGen<Stream>::nIndent() {
    return streamStack_.back().nIndent_;
}

template <class Stream>
std::string CodeGen<Stream>::toString(
    const std::function<std::string(const Stream &)> &action) {
    std::string ret;
    for (auto &&stream : poppedStream_) {
        ret += action(stream);
    }
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        ret += action(*it);
    }
    return ret;
}

} // namespace freetensor

#endif // DETAIL_CODE_GEN_H
