#include <cctype>

#include <debug.h>
#include <serialize/mangle.h>

namespace freetensor {

std::string mangle(const std::string &name) {
    std::string code;
    code.reserve(name.size() + 1);
    code += '_'; // Prepend an underscore to avoid conflicts with keywords in
                 // target languages
    for (char c : name) {
        if (isalnum(c)) {
            code += c;
        } else if (c == '_') {
            code += "__";
        } else {
            code += "_" + std::to_string((int)c) + "_";
        }
    }
    return code;
}

std::string unmangle(const std::string &code) {
    std::string name;
    name.reserve(code.size());
    for (size_t i = 1; i < code.size(); i++) {
        if (code[i] == '_') {
            if (code[++i] == '_') {
                name += '_';
            } else {
                std::string num;
                for (; code[i] != '_'; i++) {
                    num += code[i];
                }
                name += char(std::stoi(num));
            }
        } else {
            name += code[i];
        }
    }
    return name;
}

} // namespace freetensor
