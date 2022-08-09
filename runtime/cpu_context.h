#ifndef CPU_CONTEXT_H
#define CPU_CONTEXT_H

#include "context.h"

class CPUContext : public Context {};

extern "C" typedef CPUContext *CPUContext_t;

#endif // CPU_CONTEXT_H
