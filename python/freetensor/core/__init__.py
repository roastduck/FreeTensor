import os
from freetensor_ffi import AccessType, MemType, DataType, ASTNodeType, TargetType
from freetensor_ffi import InvalidSchedule, InvalidProgram, DriverError, AssertAlwaysFalse
from freetensor_ffi import dump_ast, load_ast

from .context import pop_ast
from .expr import *
from .stmt import (
    VarDef,
    For,
    If,
    Else,
    Alloc,
    Free,
    Assert,
    MarkNid,
    NamedScope,
    Invoke,
    Eval,
    Any,
    Func,
)
from .analyze import *
from .autograd import *
from .passes import *
from .schedule import *
from .codegen import NativeCode, codegen
from .driver import *
from .config import *

from .transformer import (transform, inline, empty, var, capture_var, metadata,
                          Var, StagingError, StagedAssignable, StagedIterable,
                          StagedPredicate, StagedTypeAnnotation, dynamic_range,
                          static_range)

from .meta import *
from .auto_schedule import *
from .optimize import optimize

from .task_scheduler import TaskScheduler
