import os

import freetensor_ffi as ffi
from freetensor_ffi import (ASTNodeType, InvalidSchedule, InvalidAutoGrad,
                            InvalidProgram, DriverError, InvalidIO,
                            SymbolNotFound, AssertAlwaysFalse, ParserError)

from .context import pop_ast, pop_ast_and_user_grads, StmtRange
from .expr import *
from .stmt import (VarDef, For, If, Else, Alloc, Free, Assert, MarkLabel,
                   NamedScope, Invoke, Eval, Any, Func, MarkVersion,
                   UserGradStaged)
from .analyze import *
from .autograd import *
from .passes import *
from .schedule import *
from .codegen import NativeCode, codegen
from .driver import *
from .config import *
from .serialize import *

from .frontend import (transform, inline, empty, var, capture_var, Var,
                       dynamic_range, static_range, push_for_backward, UserGrad)
from .staging import (StagingError, StagedAssignable, StagedIterable,
                      StagedPredicate, StagedTypeAnnotation)

from .meta import *
from .optimize import optimize, optimize_to_pytorch
