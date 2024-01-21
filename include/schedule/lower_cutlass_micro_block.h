#ifndef FREE_TENSOR_LOWER_CUTLASS_MICRO_BLOCK_H
#define FREE_TENSOR_LOWER_CUTLASS_MICRO_BLOCK_H

#include <stmt.h>

namespace freetensor {

Stmt lowerCutlassMicroBlock(const Stmt &ast, const ID &matMulId,
                            const ID &defIdC,
                            const std::vector<bool> &dimsCBatch,
                            const std::vector<bool> &dimsCM,
                            const std::vector<bool> &dimsCN);

}

#endif // FREE_TENSOR_LOWER_CUTLASS_MICRO_BLOCK_H
