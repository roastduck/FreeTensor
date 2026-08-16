#include <subprocess.h>

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

#include <except.h>
#include <serialize/load_ast.h>
#include <serialize/print_ast.h>

namespace freetensor {

namespace {

constexpr int INFO_FD = 3;

using json = nlohmann::json;

std::string shellQuoteForMessage(const std::string &s) {
    std::string ret = "'";
    for (char c : s) {
        if (c == '\'') {
            ret += "'\\''";
        } else {
            ret += c;
        }
    }
    ret += "'";
    return ret;
}

void closeFd(int fd) {
    if (fd >= 0) {
        while (close(fd) < 0 && errno == EINTR) {
        }
    }
}

void setNonblock(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
        throw SubprocessError(FT_MSG << "Failed to set non-blocking pipe: "
                                     << strerror(errno));
    }
}

void makePipe(int fds[2]) {
    if (pipe(fds) < 0) {
        throw SubprocessError(FT_MSG << "Failed to create pipe: "
                                     << strerror(errno));
    }
}

template <size_t N>
void dupFdsToTargets(const std::array<int, N> &sourceFds,
                     const std::array<int, N> &targetFds) {
    std::array<int, N> safeSourceFds = sourceFds;
    int firstSafeFd = *std::max_element(targetFds.begin(), targetFds.end()) + 1;

    for (auto &fd : safeSourceFds) {
        if (std::find(targetFds.begin(), targetFds.end(), fd) !=
            targetFds.end()) {
            fd = fcntl(fd, F_DUPFD, firstSafeFd);
            if (fd < 0) {
                _exit(127);
            }
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (dup2(safeSourceFds[i], targetFds[i]) < 0) {
            _exit(127);
        }
    }

    for (size_t i = 0; i < N; i++) {
        if (std::find(targetFds.begin(), targetFds.end(), sourceFds[i]) ==
            targetFds.end()) {
            closeFd(sourceFds[i]);
        }
        if (safeSourceFds[i] != sourceFds[i] &&
            std::find(targetFds.begin(), targetFds.end(), safeSourceFds[i]) ==
                targetFds.end()) {
            closeFd(safeSourceFds[i]);
        }
    }
}

std::string executablePath() {
    if (const char *env = getenv("FREETENSOR_TRANSFORM_EXECUTABLE")) {
        if (*env) {
            return env;
        }
    }
    return "freetensor-transform";
}

std::string timeoutSeconds(const std::optional<double> &timeout) {
    std::ostringstream os;
    os.precision(17);
    os << *timeout << "s";
    return os.str();
}

std::vector<std::string> makeArgvStrings(const std::string &name,
                                         const std::vector<std::string> &args,
                                         const std::optional<double> &timeout) {
    std::vector<std::string> ret;
    auto exe = executablePath();
    if (timeout.has_value()) {
        ret.emplace_back("timeout");
        ret.emplace_back("--kill-after=1s");
        ret.emplace_back(timeoutSeconds(timeout));
        ret.emplace_back(exe);
    } else {
        ret.emplace_back(exe);
    }
    ret.emplace_back(name);
    ret.insert(ret.end(), args.begin(), args.end());
    ret.emplace_back("--info-fd");
    ret.emplace_back(std::to_string(INFO_FD));
    return ret;
}

std::string commandForMessage(const std::vector<std::string> &argv) {
    std::string ret;
    for (auto &&arg : argv) {
        if (!ret.empty()) {
            ret += " ";
        }
        ret += shellQuoteForMessage(arg);
    }
    return ret;
}

[[noreturn]] void rethrowChildException(const json &exception) {
    if (!exception.contains("type") || !exception["type"].is_string()) {
        throw SubprocessError(FT_MSG
                              << "Child exception status does not have a "
                              << "string type field: " << exception.dump());
    }
    if (!exception.contains("message") || !exception["message"].is_string()) {
        throw SubprocessError(
            FT_MSG << "Child exception status does not have a string message "
                   << "field: " << exception.dump());
    }

    auto type = exception["type"].get<std::string>();
    auto message = exception["message"].get<std::string>();

    if (type == "AssertAlwaysFalse") {
        throw AssertAlwaysFalse(message);
    } else if (type == "InvalidSchedule") {
        throw InvalidSchedule(message);
    } else if (type == "InvalidAutoGrad") {
        throw InvalidAutoGrad(message);
    } else if (type == "DriverError") {
        throw DriverError(message);
    } else if (type == "InvalidIO") {
        throw InvalidIO(message);
    } else if (type == "InvalidProgram") {
        throw InvalidProgram(message);
    } else if (type == "SymbolNotFound") {
        throw SymbolNotFound(message);
    } else if (type == "ParserError") {
        throw ParserError(message);
    } else if (type == "UnexpectedQueryResult") {
        throw UnexpectedQueryResult(message);
    } else if (type == "Error") {
        throw Error(message);
    } else {
        throw SubprocessError(message.empty()
                                  ? "Unknown child exception type " + type
                                  : "Unknown child exception type " + type +
                                        ": " + message);
    }
}

std::optional<json> parseInfoEnvelope(const std::string &envelopeText,
                                      const std::vector<std::string> &argv) {
    if (envelopeText.empty()) {
        throw SubprocessError(
            FT_MSG << "Subprocess transformation did not report status on fd "
                   << INFO_FD << ": " << commandForMessage(argv));
    }

    json envelope;
    try {
        envelope = json::parse(envelopeText);
    } catch (const json::exception &e) {
        throw SubprocessError(
            FT_MSG << "Subprocess transformation reported malformed status on "
                   << "fd " << INFO_FD << ": " << e.what() << "\n"
                   << envelopeText);
    }

    if (!envelope.contains("ok") || !envelope["ok"].is_boolean()) {
        throw SubprocessError(
            FT_MSG << "Subprocess transformation reported status without a "
                   << "boolean ok field: " << envelope.dump());
    }

    if (!envelope["ok"].get<bool>()) {
        if (!envelope.contains("exception") ||
            !envelope["exception"].is_object()) {
            throw SubprocessError(
                FT_MSG << "Subprocess transformation reported failure without "
                       << "exception information: " << envelope.dump());
        }
        rethrowChildException(envelope["exception"]);
    }

    if (envelope.contains("info")) {
        return envelope["info"];
    }
    return std::nullopt;
}

} // namespace

bool shouldRunInSubprocess(const std::optional<bool> &asSubprocess,
                           const std::optional<double> &timeout) {
    if (timeout.has_value() && asSubprocess.has_value() && !*asSubprocess) {
        throw SubprocessError(
            "timeout implies as_subprocess=True, but as_subprocess=False was "
            "specified");
    }
    return timeout.has_value() || asSubprocess.value_or(false);
}

SubprocessResult runTransformSubprocess(const std::string &name,
                                        const AST &input,
                                        const std::vector<std::string> &args,
                                        const std::optional<double> &timeout) {
    int stdinPipe[2], stdoutPipe[2], stderrPipe[2], infoPipe[2];
    makePipe(stdinPipe);
    makePipe(stdoutPipe);
    makePipe(stderrPipe);
    makePipe(infoPipe);

    auto argvStrings = makeArgvStrings(name, args, timeout);
    std::vector<char *> argv;
    argv.reserve(argvStrings.size() + 1);
    for (auto &arg : argvStrings) {
        argv.push_back(arg.data());
    }
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) {
        throw SubprocessError(FT_MSG << "Failed to fork: " << strerror(errno));
    }

    if (pid == 0) {
        constexpr std::array targetFds = {STDIN_FILENO, STDOUT_FILENO,
                                          STDERR_FILENO, INFO_FD};
        std::array sourceFds = {stdinPipe[0], stdoutPipe[1], stderrPipe[1],
                                infoPipe[1]};
        std::array unusedFds = {stdinPipe[1], stdoutPipe[0], stderrPipe[0],
                                infoPipe[0]};

        dupFdsToTargets(sourceFds, targetFds);
        for (int fd : unusedFds) {
            if (std::find(targetFds.begin(), targetFds.end(), fd) ==
                targetFds.end()) {
                closeFd(fd);
            }
        }

        execvp(argv[0], argv.data());
        std::cerr << "Failed to exec " << argvStrings[0] << ": "
                  << strerror(errno) << std::endl;
        _exit(127);
    }

    closeFd(stdinPipe[0]);
    closeFd(stdoutPipe[1]);
    closeFd(stderrPipe[1]);
    closeFd(infoPipe[1]);

    setNonblock(stdinPipe[1]);
    setNonblock(stdoutPipe[0]);
    setNonblock(stderrPipe[0]);
    setNonblock(infoPipe[0]);

    std::string inputText = dumpAST(input);
    size_t inputPos = 0;
    std::string stdoutText, stderrText, infoText;
    bool stdinOpen = true, stdoutOpen = true, stderrOpen = true,
         infoOpen = true;

    while (stdinOpen || stdoutOpen || stderrOpen || infoOpen) {
        std::vector<pollfd> fds;
        enum { FD_STDIN, FD_STDOUT, FD_STDERR, FD_INFO };
        std::vector<int> kinds;

        if (stdinOpen) {
            fds.push_back({stdinPipe[1], POLLOUT, 0});
            kinds.push_back(FD_STDIN);
        }
        if (stdoutOpen) {
            fds.push_back({stdoutPipe[0], POLLIN, 0});
            kinds.push_back(FD_STDOUT);
        }
        if (stderrOpen) {
            fds.push_back({stderrPipe[0], POLLIN, 0});
            kinds.push_back(FD_STDERR);
        }
        if (infoOpen) {
            fds.push_back({infoPipe[0], POLLIN, 0});
            kinds.push_back(FD_INFO);
        }

        int ret = poll(fds.data(), fds.size(), -1);
        if (ret < 0) {
            if (errno == EINTR) {
                continue;
            }
            throw SubprocessError(FT_MSG << "Failed to poll subprocess pipes: "
                                         << strerror(errno));
        }

        for (size_t i = 0; i < fds.size(); i++) {
            auto revents = fds[i].revents;
            if (!revents) {
                continue;
            }
            int kind = kinds[i];
            if (kind == FD_STDIN) {
                if (inputPos < inputText.size()) {
                    ssize_t n = write(stdinPipe[1], inputText.data() + inputPos,
                                      inputText.size() - inputPos);
                    if (n > 0) {
                        inputPos += n;
                    } else if (n < 0 && errno != EAGAIN &&
                               errno != EWOULDBLOCK && errno != EINTR &&
                               errno != EPIPE) {
                        throw SubprocessError(
                            FT_MSG << "Failed to write to subprocess stdin: "
                                   << strerror(errno));
                    }
                }
                if (inputPos >= inputText.size() ||
                    (revents & (POLLERR | POLLHUP | POLLNVAL))) {
                    closeFd(stdinPipe[1]);
                    stdinOpen = false;
                }
            } else {
                int fd = kind == FD_STDOUT   ? stdoutPipe[0]
                         : kind == FD_STDERR ? stderrPipe[0]
                                             : infoPipe[0];
                std::string &buf = kind == FD_STDOUT   ? stdoutText
                                   : kind == FD_STDERR ? stderrText
                                                       : infoText;
                char tmp[8192];
                while (true) {
                    ssize_t n = read(fd, tmp, sizeof(tmp));
                    if (n > 0) {
                        buf.append(tmp, n);
                    } else if (n == 0) {
                        if (kind == FD_STDOUT) {
                            stdoutOpen = false;
                        } else if (kind == FD_STDERR) {
                            stderrOpen = false;
                        } else {
                            infoOpen = false;
                        }
                        closeFd(fd);
                        break;
                    } else if (errno == EAGAIN || errno == EWOULDBLOCK ||
                               errno == EINTR) {
                        break;
                    } else {
                        throw SubprocessError(
                            FT_MSG << "Failed to read from subprocess pipe: "
                                   << strerror(errno));
                    }
                }
                if (revents & (POLLERR | POLLHUP | POLLNVAL)) {
                    if (kind == FD_STDOUT) {
                        stdoutOpen = false;
                        closeFd(stdoutPipe[0]);
                    } else if (kind == FD_STDERR) {
                        stderrOpen = false;
                        closeFd(stderrPipe[0]);
                    } else {
                        infoOpen = false;
                        closeFd(infoPipe[0]);
                    }
                }
            }
        }
    }

    int status;
    while (waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw SubprocessError(FT_MSG << "Failed to wait for subprocess: "
                                         << strerror(errno));
        }
    }

    int exitCode = -1;
    if (WIFEXITED(status)) {
        exitCode = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        exitCode = 128 + WTERMSIG(status);
    }

    if (timeout.has_value() && exitCode == 124) {
        throw SubprocessError(FT_MSG << "Transformation " << name
                                     << " timed out after " << *timeout
                                     << " seconds");
    }
    std::optional<json> transformationInfo =
        parseInfoEnvelope(infoText, argvStrings);

    if (exitCode != 0) {
        throw SubprocessError(
            FT_MSG << "Subprocess transformation failed with exit "
                   << "code " << exitCode << " after reporting success: "
                   << commandForMessage(argvStrings)
                   << (stderrText.empty() ? "" : "\nstderr:\n") << stderrText);
    }

    SubprocessResult ret;
    ret.ast_ = loadAST(stdoutText);
    ret.info_ = transformationInfo;
    ret.stderr_ = stderrText;
    return ret;
}

} // namespace freetensor
