#ifndef CONFIG_H
#define CONFIG_H

#include <string>

namespace ir {

class Config {
    static bool prettyPrint_;

  public:
    static std::string withMKL();

    static void setPrettyPrint(bool pretty = true) { prettyPrint_ = pretty; }
    static bool prettyPrint() { return prettyPrint_; }
};

} // namespace ir

#endif // CONFIG_H
