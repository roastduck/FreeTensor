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
void CodeGen<Stream>::markDefBuffer(const std::string &name,
                                    const Ref<Buffer> &buffer) {
    var2Stream_[name] = streamStack_.back().name_;
    buffers_[name] = buffer;
}

template <class Stream>
void CodeGen<Stream>::markUseBuffer(const std::string &name) {
    auto &&stream = var2Stream_.at(name);
    auto &&buffer = buffers_.at(name);
    for (auto it = streamStack_.rbegin(); it != streamStack_.rend(); it++) {
        if (it->name_ == stream) {
            break;
        }
        it->useBuffers_[name] = buffer;
    }
}

template <class Stream>
void CodeGen<Stream>::markUndefBuffer(const std::string &name) {
    var2Stream_.erase(name);
    buffers_.erase(name);
}

template <class Stream>
void CodeGen<Stream>::markDefIter(const std::string &name) {
    var2Stream_[name] = streamStack_.back().name_;
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

template <class Stream>
void CodeGen<Stream>::markUndefIter(const std::string &name) {
    var2Stream_.erase(name);
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
