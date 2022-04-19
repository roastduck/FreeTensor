parser grammar ast_parser;

options {
    tokenVocab = ast_lexer;
}

@parser::postinclude {
    #include <iostream>
    #include <string>
    #include <vector>

    #include <stmt.h>
    #include <func.h>
}

program returns [AST node]
    : func EOF {
        $node = $func.node;
    }
    | stmts EOF {
        $node = $stmts.node;
    }
    ;

// -------------------- type --------------------
mtype returns [MemType type]
    : '['
        (BYVALUE {$type = MemType::ByValue;}
        |CPU {$type = MemType::CPU;}
        |GPUGlobal {$type = MemType::GPUGlobal;}
        |GPUShared {$type = MemType::GPUShared;}
        |GPULocal {$type = MemType::GPULocal;}
        |GPUWarp {$type = MemType::GPUWarp;})
        ']';

atype returns [AccessType type]
    : '[' (IN {$type = AccessType::Input;}
            |OUT {$type = AccessType::Output;}
            |INOUT {$type = AccessType::InOut;}
            |CACHE {$type = AccessType::Cache;}) ']';

dtype returns [DataType type]
    : (F32 {$type = DataType::Float32;}
    |F64 {$type = DataType::Float64;}
    |I32 {$type = DataType::Int32;}
    |BOOL {$type = DataType::Bool;}
    |CUSTOM {$type = DataType::Custom;}
    |VOID {$type = DataType::Void;});

func returns [Func node]
    : FUNC '(' var_params ')' (RARROW)? '{' stmts '}'
      {
        std::vector<std::pair<std::string, DataType>> returns;
        Stmt body;
        // TODO name, returns, closure
        $node = makeFunc("name", $var_params.vec, {}, $stmts.node, {});
      }
    ;

// -------------------- STMT --------------------
// All statements default to without ID, and stmt will handle the ID
stmts returns [Stmt node]
    : stmt {
        std::vector<Stmt> stmts;
        stmts.emplace_back($stmt.node);
    } (newStmt=stmt {
        stmts.emplace_back($newStmt.node);
    })+ {
        $node = makeStmtSeq(ID(), std::move(stmts));
    }
    | stmt {
        $node = $stmt.node;
    };

stmt returns [Stmt node]
    : '{' '}'
      {
        $node = makeStmtSeq(ID(), {});
      }
    | stmtWithoutID
      {
        $node = $stmtWithoutID.node;
      }
    | Var COLON stmtWithoutID
      {
        $node = $stmtWithoutID.node;
        $node->setId($Var.text);
      }
    | Var COLON '{' stmts '}'
      {
        $node = $stmts.node;
        $node->setId($Var.text);
      }
    ;

stmtWithoutID returns [Stmt node]
    : store
      {
        $node = $store.node;
      }
    | reduceTo
      {
        $node = $reduceTo.node;
      }
    | varDef
      {
        $node = $varDef.node;
      }
    | forNode
      {
        $node = $forNode.node;
      }
    | ifNode
      {
        $node = $ifNode.node;
      }
    ;

store returns [Stmt node]
    : Var indices '=' expr {
        std::string var = $Var.text;
        std::vector<Expr> indices = $indices.exprs;
        $node = makeStore(ID(), var, std::move(indices), $expr.node);
    };

reduceTo returns [Stmt node]
    @init {
        bool atomic = false;
        ReduceOp op;
    }
    : (ATOMIC {atomic = true;})?
        Var indices
        (PLUSEQ {op = ReduceOp::Add;}
        |STAREQ {op = ReduceOp::Mul;}
        |MINEQ {op = ReduceOp::Min;}
        |MAXEQ {op = ReduceOp::Max;})
        expr {
            $node = makeReduceTo(ID(), $Var.text, $indices.exprs, op, $expr.node, atomic);
        }
    ;

load returns [Expr node]
    : Var indices
      {
        $node = makeLoad($Var.text, $indices.exprs);
      }
    ;

varDef returns [Stmt node]
    @init {
        Expr sizeLim = nullptr;
        bool pinned = false;
    }
    : atype mtype Var COLON dtype shape
        (SIZELIM '=' expr {sizeLim = $expr.node;})?
        (PINNED {pinned = true;})?
        '{'
        stmts
        '}' {
            AccessType atype = $atype.type;
            MemType mtype = $mtype.type;
            std::string name = $Var.text;
            DataType dtype = $dtype.type;
            std::vector<Expr> shape = $shape.vec;
            Ref<Tensor> t = makeTensor(std::move(shape), dtype);
            Ref<Buffer> b = makeBuffer(std::move(t), atype, mtype);
            Expr sizeLim = nullptr;
            Stmt body = $stmts.node;
            $node = makeVarDef(ID(), name, std::move(b), std::move(sizeLim), body, pinned);
        };

forNode returns [Stmt node]
    @init {
        auto property = Ref<ForProperty>::make();
    }
    : (NO_DEPS '=' Var
      {
        property->noDeps_.emplace_back($Var.text);
      }
      (COMMA newVar=Var
      {
        property->noDeps_.emplace_back($newVar.text);
      }
      )*)?
      (PARALLEL '=' (OPENMP
      {
        property = property->withParallel(OpenMPScope{});
      }
      | CUDASTREAM
      {
        property = property->withParallel(CUDAStreamScope{});
      }
      | {CUDAScope::Level level;}
          (SCOPE_BLOCK {level = CUDAScope::Level::Block;}
          | SCOPE_THREAD {level = CUDAScope::Level::Thread;})
          {CUDAScope::Dim dim;}
          (DOTX {dim = CUDAScope::Dim::X;}
          | DOTY {dim = CUDAScope::Dim::Y;}
          | DOTZ {dim = CUDAScope::Dim::Z;})
      {
        property = property->withParallel(CUDAScope{level, dim});
      }
      ))?
        // TODO: reduction
        (
            UNROLL {property = property->withUnroll();}
        )? // unroll
        (
            VECTORIZE {property = property->withVectorize();}
        )? // vectorize
        (
            PREFERLIBS {property = property->withPreferLibs();}
        )? // preferLibs
        FOR Var IN begin=expr COLON end=expr COLON step=expr COLON len=expr '{'
        stmts
        '}' {
            $node = makeFor(ID(), $Var.text, $begin.node, $end.node, $step.node, $len.node, std::move(property), $stmts.node);
        };

ifNode returns [Stmt node]
    : IF '(' cond=expr ')' '{' thenCase=stmts '}'
      {
        $node = makeIf(ID(), $cond.node, $thenCase.node);
      }
    | IF '(' cond=expr ')' '{' thenCase=stmts '}' ELSE '{' elseCase=stmts '}'
      {
        $node = makeIf(ID(), $cond.node, $thenCase.node, $elseCase.node);
      }
    ;

// -------------------- EXPR --------------------
 // TODO: Intrinsic
expr returns [Expr node]
    : load
      {
        $node = $load.node;
      }
    | intConst
      {
        $node = $intConst.node;
      }
    | floatConst
      {
        $node = $floatConst.node;
      }
    | boolConst
      {
        $node = $boolConst.node;
      }
    | Var
      {
        $node = makeVar($Var.text);
      }
    | '(' expr0=expr QUESTION expr1=expr COLON expr2=expr ')'
      {
        $node = makeIfExpr($expr0.node, $expr1.node, $expr2.node);
      }
    | dtype '(' expr ')'
      {
        $node = makeCast($expr.node, $dtype.type);
      }
    | NOT expr
      {
        $node = makeLNot($expr.node);
      }
    | SQRT '(' expr ')'
      {
        $node = makeSqrt($expr.node);
      }
    | EXP '(' expr ')'
      {
        $node = makeExp($expr.node);
      }
    | ABS '(' expr ')'
      {
        $node = makeAbs($expr.node);
      }
    | SIGMOID '(' expr ')'
      {
        $node = makeSigmoid($expr.node);
      }
    | TANH '(' expr ')'
      {
        $node = makeTanh($expr.node);
      }
    | '(' expr ')' SQUARE
      {
        $node = makeSquare($expr.node);
      }
    | FLOOR '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeFloorDiv($expr0.node, $expr1.node);
      }
    | CEIL '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeFloorDiv($expr0.node, $expr1.node);
      }
    | ROUNDTO0 '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeFloorDiv($expr0.node, $expr1.node);
      }
    | FLOOR '(' expr ')'
      {
        $node = makeFloor($expr.node);
      }
    | CEIL '(' expr ')'
      {
        $node = makeCeil($expr.node);
      }
    | expr0=expr '*' expr1=expr
      {
        $node = makeMul($expr0.node, $expr1.node);
      }
    | expr0=expr '/' expr1=expr
      {
        $node = makeRealDiv($expr0.node, $expr1.node);
      }
    | expr0=expr '%' expr1=expr
      {
        $node = makeMod($expr0.node, $expr1.node);
      }
    | expr0=expr '%%' expr1=expr
      {
        $node = makeRemainder($expr0.node, $expr1.node);
      }
    | expr0=expr '+' expr1=expr
      {
        $node = makeAdd($expr0.node, $expr1.node);
      }
    | expr0=expr '-' expr1=expr
      {
        $node = makeSub($expr0.node, $expr1.node);
      }
    | expr0=expr '<=' expr1=expr
      {
        $node = makeLE($expr0.node, $expr1.node);
      }
    | expr0=expr '<' expr1=expr
      {
        $node = makeLT($expr0.node, $expr1.node);
      }
    | expr0=expr '>=' expr1=expr
      {
        $node = makeGE($expr0.node, $expr1.node);
      }
    | expr0=expr '>' expr1=expr
      {
        $node = makeGT($expr0.node, $expr1.node);
      }
    | expr0=expr '==' expr1=expr
      {
        $node = makeEQ($expr0.node, $expr1.node);
      }
    | expr0=expr '!=' expr1=expr
      {
        $node = makeNE($expr0.node, $expr1.node);
      }
    | expr0=expr '&&' expr1=expr
      {
        $node = makeLAnd($expr0.node, $expr1.node);
      }
    | expr0=expr '||' expr1=expr
      {
        $node = makeLOr($expr0.node, $expr1.node);
      }
    | MIN '(' expr0=expr COMMA expr1=expr ')'
      {
        $node = makeMin($expr0.node, $expr1.node);
      }
    | MAX '(' expr0=expr COMMA expr1=expr ')'
      {
        $node = makeMax($expr0.node, $expr1.node);
      }
    | '(' expr ')'
      {
        $node = $expr.node;
      }
    | INTRINSIC '(' DQUOTE DQUOTE ')'
    ;


// -------------------- PARAMETERS --------------------
shape returns [std::vector<Expr> vec]
    : '[' expr
      {
        $vec.emplace_back($expr.node);
      }
      (COMMA newExprNode=expr
      {
        $vec.emplace_back($newExprNode.node);
      }
      )* ']'
    |
      '[' ']'
    ;

var_params returns [std::vector<std::string> vec]
    : var=Var  {
        $vec.push_back(std::string($var.text));
    } (COMMA var1=Var {$vec.push_back(std::string($var1.text));})*;

indices returns [std::vector<Expr> exprs]
    : '[' expr
      {
        $exprs.emplace_back($expr.node);
      }
      (COMMA newExpr=expr
      {
        $exprs.emplace_back($newExpr.node);
      }
      )* ']'
    | '[' ']'
    ;

// -------------------- CONST --------------------
intConst returns [Expr node]
    : Integer {
        const char *s = $Integer.text.c_str();
        $node = makeIntConst(std::atoll(s));
    };

floatConst returns [Expr node]
    : Float {
        auto s = $Float.text.c_str();
        $node = makeFloatConst(std::atof(s));
    };

boolConst returns [Expr node]
    @init {bool val;}
    : (
        TRUE {val = true;}
        |FALSE {val = false;}
    ) {$node = makeBoolConst(val);};
