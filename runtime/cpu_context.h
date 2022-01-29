#ifndef CPU_CONTEXT_H
#define CPU_CONTEXT_H

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sys/resource.h> // rlimit

#include "context.h"

class CPUContext : public Context {
    int64_t curStackSize_ = 0;

  public:
    void setStackLim(int64_t bytes) {
        if (bytes > curStackSize_) {
            struct rlimit rlim;
            auto err = getrlimit(RLIMIT_STACK, &rlim);
            if (err != 0) {
                std::cerr << "Error getting rlimit" << std::endl;
                exit(-1);
            }
            if ((rlim_t)bytes > rlim.rlim_cur) {
                rlim.rlim_cur = bytes;
                err = setrlimit(RLIMIT_STACK, &rlim);
                if (err != 0) {
                    std::cerr << "Error getting rlimit" << std::endl;
                    exit(-1);
                }
            }
            curStackSize_ = bytes;
        }
    }
};

extern "C" typedef CPUContext *CPUContext_t;

#endif // CPU_CONTEXT_H
