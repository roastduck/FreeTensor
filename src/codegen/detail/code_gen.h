#ifndef DETAIL_CODE_GEN_H
#define DETAIL_CODE_GEN_H

#include <codegen/code_gen.h>

namespace ir {

template <class Stream> CodeGen<Stream>::CodeGen() { pushStream("default"); }

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
    for (int i = 0, iEnd = nIndent(); i < iEnd; i++) {
        os() << "  ";
    }
}

template <class Stream>
void CodeGen<Stream>::markDef(const std::string &name,
                              const Ref<Buffer> &buffer) {
    vars_[name] = std::make_pair(streamStack_.back().name_, buffer);
}

template <class Stream> void CodeGen<Stream>::markUse(const std::string &name) {
    auto &&def = vars_.at(name);
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        if (it->name_ == def.first) {
            break;
        }
        it->uses_[name] = def.second;
    }
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

} // namespace ir

#endif // DETAIL_CODE_GEN_H
