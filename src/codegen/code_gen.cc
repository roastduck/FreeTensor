#include <codegen/code_gen.h>

namespace ir {

CodeGen::CodeGen() { pushStream("default"); }

void CodeGen::beginBlock() {
    os() << "{" << std::endl;
    nIndent()++;
}

void CodeGen::endBlock() {
    nIndent()--;
    makeIndent();
    os() << "}" << std::endl;
}

void CodeGen::makeIndent() {
    for (int i = 0, iEnd = nIndent(); i < iEnd; i++) {
        os() << "  ";
    }
}

void CodeGen::markDef(const std::string &name, const Ref<Buffer> &buffer) {
    vars_[name] = buffer;
}

void CodeGen::markUse(const std::string &name) {
    for (Stream &item : streamStack_) {
        item.uses_[name] = vars_.at(name);
    }
}

void CodeGen::pushStream(const std::string &name) {
    streamStack_.emplace_back();
    streamStack_.back().name_ = name;
}

void CodeGen::popStream() {
    poppedStream_.emplace_back(std::move(streamStack_.back()));
    streamStack_.pop_back();
}

std::ostringstream &CodeGen::os() { return streamStack_.back().os_; }

int &CodeGen::nIndent() { return streamStack_.back().nIndent_; }

std::string
CodeGen::toString(const std::function<std::string(const Stream &)> &action) {
    std::string ret;
    for (auto &&stream : poppedStream_) {
        ret += action(stream);
    }
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        ret += action(*it);
    }
    return ret;
}

} // namespace ir

