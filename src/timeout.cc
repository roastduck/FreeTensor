#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <except.h>
#include <timeout.h>

namespace freetensor {

static void readUntilSucess(int fd, void *buf, size_t count) {
    while (count > 0) {
        auto thisCount = read(fd, buf, count);
        count -= thisCount;
        buf = ((std::byte *)buf) + thisCount;
    }
}

std::vector<std::byte>
timeout(const std::function<std::vector<std::byte>()> &func, int seconds) {
    // Crate a pipe to pass the result. Data format: a 8-byte size followed by
    // data bytes.
    int fd[2];
    if (pipe(fd) == -1) {
        ERROR("Failed to create a pipe");
    }

    pid_t pid = fork();
    if (pid == -1) {
        ERROR("Failed to fork a new process");
    } else if (pid == 0) { // Child process
        close(fd[0]);      // Close the read end of the pipe
        auto bytes = func();
        size_t size = bytes.size();
        write(fd[1], &size, sizeof(size_t));
        write(fd[1], bytes.data(), size);
        close(fd[1]); // Close the write end of the pipe
        exit(EXIT_SUCCESS);
    } else {          // Parent process
        close(fd[1]); // Close the write end of the pipe
        std::vector<std::byte> result;
        struct pollfd pfd;
        pfd.fd = fd[0];
        pfd.events = POLLIN;
        int ret = poll(&pfd, 1,
                       seconds * 1000); // Wait for the child process to finish
                                        // or for the timeout to expire
        if (ret == -1) {
            close(fd[0]);
            ERROR("poll returns an error");
        } else if (ret == 0) { // Timeout expired, so kill the child process
            WARNING("Function timed out after " + std::to_string(seconds) +
                    " seconds");
            kill(pid, SIGKILL);
        } else { // Data is ready to be read from the pipe
            size_t size;
            readUntilSucess(fd[0], &size, sizeof(size_t));
            result.resize(size);
            readUntilSucess(fd[0], result.data(), size);
        }
        close(fd[0]); // Close the read end of the pipe
        return result;
    }
}

} // namespace freetensor
