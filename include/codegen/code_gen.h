#ifndef CODE_GEN_H
#define CODE_GEN_H

#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <visitor.h>

namespace ir {

struct CodeGenStream {
    std::string name_;
    std::ostringstream os_;
    int nIndent_ = 0;
    std::unordered_map<std::string, Ref<Buffer>> useBuffers_;
    std::unordered_set<std::string> useIters_;
};

template <class Stream> class CodeGen : public Visitor {
  protected:
    std::vector<Stream> streamStack_, poppedStream_;

    std::unordered_map<std::string, Ref<Buffer>> buffers_; // var name -> buffer
    std::unordered_map<std::string, std::string>
        var2Stream_; // var name -> stream name

    void makeIndent();

    template <class T> void printList(T &&list) {
        for (size_t i = 0, iEnd = list.size(); i < iEnd; i++) {
            (*this)(list[i]);
            os() << (i < iEnd - 1 ? ", " : "");
        }
    }

    CodeGen();

    void markDefBuffer(const std::string &name, const Ref<Buffer> &buffer);
    void markUseBuffer(const std::string &name);
    void markUndefBuffer(const std::string &name);

    void markDefIter(const std::string &name);
    void markUseIter(const std::string &name);
    void markUndefIter(const std::string &name);

    void pushStream(const std::string &name);
    void popStream();

    std::ostream &os();
    int &nIndent();

  public:
    void beginBlock();
    void endBlock();

    /**
     * Dump all streams to a string
     *
     * @param action : callback(stream). Do more modification to a stream.
     * Function prelude and finale can be added here
     */
    std::string
    toString(const std::function<std::string(const Stream &)> &action);
};

} // namespace ir

#endif // CODE_GEN_H
