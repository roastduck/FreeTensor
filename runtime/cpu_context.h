#ifndef FREE_TENSOR_CPU_CONTEXT_H
#define FREE_TENSOR_CPU_CONTEXT_H

#include "context.h"

class CPUContext : public Context {};

extern "C" typedef CPUContext *CPUContext_t;

#endif // FREE_TENSOR_CPU_CONTEXT_H
