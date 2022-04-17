parser grammar MyParser;

options {
    tokenVocab = MyLexer;
}

@parser::postinclude {
    #include <iostream>
    #include <string>
    #include <vector>
    #include "../../include/stmt.h"
    #include "../../include/func.h"
}

program returns [ir::Func node]
    : func EOF {
            $node = $func.node;
        };

// ---------- ---------- type ---------- ----------
mtype returns [ir::MemType type]
    : LBRACK
        (BYVALUE {$type = ir::MemType::ByValue;}
        |CPU {$type = ir::MemType::CPU;}
        |GPUGlobal {$type = ir::MemType::GPUGlobal;}
        |GPUShared {$type = ir::MemType::GPUShared;}
        |GPULocal {$type = ir::MemType::GPULocal;})
        |GPUWarp {$type = ir::MemType::GPUWarp;}
        RBRACK;

atype returns [ir::AccessType type]
    : LBRACK (IN {$type = ir::AccessType::Input;}
            |OUT {$type = ir::AccessType::Output;}
            |INOUT {$type = ir::AccessType::InOut;}
            |CACHE {$type = ir::AccessType::Cache;}) RBRACK;

dtype returns [ir::DataType type]
    : (F32 {$type = ir::DataType::Float32;}
    |F64 {$type = ir::DataType::Float64;}
    |I32 {$type = ir::DataType::Int32;}
    |BOOL {$type = ir::DataType::Bool;}
    |CUSTOM {$type = ir::DataType::Custom;}
    |VOID {$type = ir::DataType::Void;});

// TODO: returns
func returns [ir::Func node]
    : FUNC LPAREN var_params RPAREN (MINUS GT)? LBRACE NEWLINE
        stmtNode NEWLINE
        RBRACE {
            std::string name("name");
            std::vector <std::string> params = $var_params.vec;
            std::vector <std::pair<std::string, ir::DataType> > returns;
            ir::Stmt body;
            // $node = ir::Func::make();
            // $node = ir::makeFunc(name, std::move(params), returns, )
        };

// ---------- ---------- STMT overall ---------- ----------
// 所有make node先默认空id，在parse到id的地方再加入
// 一个Stmt  or  多个stmtSingle组成的stmtSeq
// TODO: 加一个 当且仅当有id且stmtNode为stmtSeq时加括号
stmtNode returns [ir::Stmt stmt]
    : stmtSingle {
        std::vector <ir::Stmt> stmts;
        stmts.clear();
        stmts.emplace_back($stmtSingle.stmt);
    } (newStmtSingle=stmtSingle NEWLINE{
        stmts.emplace_back($newStmtSingle.stmt);
        $stmt = ir::makeStmtSeq(ir::ID(), std::move(stmts));
    })+ {}
    | stmtSingle {
        $stmt = $stmtSingle.stmt;
    };

// 非stmtSeq的stmt  or  带id的stmt
// 这里强制id下加括号
stmtSingle returns [ir::Stmt stmt]
    : EMPTY_STMTSEQ {
        std::vector <ir::Stmt> stmts;
        stmts.clear();
        $stmt = ir::makeStmtSeq(ir::ID(), std::move(stmts));
    }
    | stmtType {
        $stmt = $stmtType.stmt;
    }
    | Var {std::string id = $Var.text;} COLON
        stmtNode NEWLINE {
            ir::ID stmtId(id);
            $stmt = $stmtNode.stmt;
            $stmt->setId(stmtId);
        }
        RBRACE;

// ---------- ---------- types of STMT ---------- ----------
stmtType returns [ir::Stmt stmt]
    : varDef {
        $stmt = $varDef.varDefNode;
    }
    | store {
        $stmt = $store.storeNode;
    }
    | reduceTo {
        $stmt = $reduceTo.reduceToNode;
    }
    | forNode {
        $stmt = $forNode.fornode;
    };

// TODO: VarDef.pinned
varDef returns [ir::Stmt varDefNode] // ir::VarDef
    @init {
        ir::Expr sizeLim = nullptr;
    }
    : atype mtype Var COLON dtype shape (
        exprNode {
            sizeLim = $exprNode.expr;
        }
    )? LBRACE NEWLINE
        stmtNode NEWLINE
        RBRACE {
            ir::ID id; // 交由 stmtSingle 统一命名
            std::cout << "varDef" << std::endl;
            ir::AccessType atype = $atype.type;
            ir::MemType mtype = $mtype.type;
            std::string name = $Var.text;
            ir::DataType dtype = $dtype.type;
            std::vector <ir::Expr> shape = $shape.vec;
            ir::Ref<ir::Tensor> t = ir::makeTensor(std::move(shape), dtype);
            ir::Ref<ir::Buffer> b = ir::makeBuffer(std::move(t), atype, mtype);
            ir::Expr sizeLim = nullptr;
            ir::Stmt body = $stmtNode.stmt;
            // $varDefNode = ir::makeVarDef(id, name, std::move(b), std::move(sizeLim), body, )
        };

store returns [ir::Stmt storeNode] // ir::Store
    : Var LBRACK indices_expr RBRACK ASSIGN exprNode {
        std::string var = $Var.text;
        std::vector<ir::Expr> indices = $indices_expr.exprs;
        auto expr = $exprNode.expr;
        $storeNode = ir::makeStore(ir::ID(), var, std::move(indices), std::move(expr));
    };

reduceTo returns [ir::Stmt reduceToNode] // ir::ReduceTo
    @init {
        bool atomic = false;
        ir::ReduceOp op;
    }
    : (ATOMIC {atomic = true;})?
        Var LBRACK indices_expr RBRACK
        (PLUSEQ {op = ir::ReduceOp::Add;}
        |STAREQ {op = ir::ReduceOp::Mul;}
        |MINEQ {op = ir::ReduceOp::Min;}
        |MAXEQ {op = ir::ReduceOp::Max;})
        exprNode {
            std::string var = $Var.text;
            std::vector<ir::Expr> indices = $indices_expr.exprs;
            auto expr = $exprNode.expr;
            $reduceToNode = ir::makeReduceTo(ir::ID(), var, std::move(indices), op, std::move(expr), atomic);
        }
    ;

// TODO: len
forNode returns [ir::Stmt fornode] // ir::ForNode
    @init {
        std::vector<std::string> no_deps;
        no_deps.clear();
        ir::ParallelScope parallel;
        bool unroll = false;
        bool vectorize = false;
        bool preferlibs = false;
    }
    :   (
            NO_DEPS ASSIGN Var{no_deps.emplace_back($Var.text);}
            (COMMA newVar=Var {no_deps.emplace_back($newVar.text);})* NEWLINE
        )? // no_deps
        (
            PARALLEL ASSIGN (
                {parallel = ir::SerialScope{};}
                | OPENMP {parallel = ir::OpenMPScope{};}
                | CUDASTREAM {parallel = ir::CUDAStreamScope{};}
                | {ir::CUDAScope::Level level;}
                    (SCOPE_BLOCK {level = ir::CUDAScope::Level::Block;}
                    | SCOPE_THREAD {level = ir::CUDAScope::Level::Thread;})
                    {ir::CUDAScope::Dim dim;}
                    (DOTX {dim = ir::CUDAScope::Dim::X;}
                    | DOTY {dim = ir::CUDAScope::Dim::Y;}
                    | DOTZ {dim = ir::CUDAScope::Dim::Z;})
                    {parallel = ir::CUDAScope{level, dim};}
            ) NEWLINE
        )? // parallel
        // TODO: reduction
        (
            UNROLL NEWLINE {unroll = true;}
        )? // unroll
        (
            VECTORIZE NEWLINE {vectorize = true;}
        )? // vectorize
        (
            PREFERLIBS NEWLINE {preferlibs = true;}
        )? // preferLibs
        FOR Var IN begin=exprNode COLON end=exprNode COLON step=exprNode LBRACE NEWLINE
        stmtNode NEWLINE
        RBRACE {
            auto begin = $begin.expr;
            auto end = $end.expr;
            auto step = $step.expr;
            auto body = $stmtNode.stmt;
        };

// ---------- ---------- EXPR ---------- ----------
exprNode returns [ir::Expr expr]
    : Var
    | intConst {$expr = $intConst.expr;}
    | floatConst {$expr = $floatConst.expr;}
    | boolConst {$expr = $boolConst.expr;}
    | store
    | load
    | LPAREN expr0=exprNode QUESTION expr1=exprNode COLON expr2=exprNode RPAREN // IfExpr
        {$expr = ir::makeIfExpr($expr0.expr, $expr1.expr, $expr2.expr);}
    | dtype LPAREN exprNode RPAREN {$expr = ir::makeCast($exprNode.expr, $dtype.type);} // Cast
    | NOT exprNode {$expr = ir::makeLNot($exprNode.expr);} // LNot
    | ( // Max, Min, Sqrt, Exp, Abs, Sigmoid, Tanh
        SQRT {$expr = ir::makeSqrt($exprNode.expr);}
        | EXP {$expr = ir::makeExp($exprNode.expr);}
        | ABS {$expr = ir::makeAbs($exprNode.expr);}
        | SIGMOID {$expr = ir::makeSigmoid($exprNode.expr);}
        | TANH {$expr = ir::makeTanh($exprNode.expr);}
        ) LPAREN exprNode RPAREN
    | LPAREN exprNode RPAREN SQUARE {$expr = ir::makeSquare($exprNode.expr);} // Square
    | ( // FloorDiv, CeilDiv, RoundTowards0Div
        FLOOR {$expr = ir::makeFloorDiv($expr0.expr, $expr1.expr);}
        | CEIL {$expr = ir::makeCeilDiv($expr0.expr, $expr1.expr);}
        | ROUNDTO0 {$expr = ir::makeRoundTowards0Div($expr0.expr, $expr1.expr);}
        ) LPAREN expr0=exprNode SLASH expr1=exprNode RPAREN
    | ( // Floor, Ceil
        FLOOR {$expr = ir::makeFloor($exprNode.expr);}
        | CEIL {$expr = ir::makeCeil($exprNode.expr);}
        ) LPAREN exprNode RPAREN
    | expr0=exprNode ( // Mul, RealDiv, Mod, Remainder
        STAR {$expr = ir::makeMul($expr0.expr, $expr1.expr);}
        | SLASH {$expr = ir::makeRealDiv($expr0.expr, $expr1.expr);}
        | PERCENT {$expr = ir::makeMod($expr0.expr, $expr1.expr);}
        | PERCENT PERCENT {$expr = ir::makeRemainder($expr0.expr, $expr1.expr);}
        ) expr1=exprNode
    | expr0=exprNode ( // Add, Sub
        PLUS {$expr = ir::makeAdd($expr0.expr, $expr1.expr);}
        | MINUS {$expr = ir::makeSub($expr0.expr, $expr1.expr);}
        ) expr1=exprNode
    | expr0=exprNode ( // <=, >=, <, >, ==, !=
        LE {$expr = ir::makeLE($expr0.expr, $expr1.expr);}
        | GE {$expr = ir::makeGE($expr0.expr, $expr1.expr);}
        | LT {$expr = ir::makeLT($expr0.expr, $expr1.expr);}
        | GT {$expr = ir::makeGT($expr0.expr, $expr1.expr);}
        ) expr1=exprNode
    | expr0=exprNode ( // ==, !=
        EQ {$expr = ir::makeEQ($expr0.expr, $expr1.expr);}
        | NE {$expr = ir::makeNE($expr0.expr, $expr1.expr);}
        ) expr1=exprNode
    | expr0=exprNode ( // &&, ||
        LAND {$expr = ir::makeLAnd($expr0.expr, $expr1.expr);}
        | LOR {$expr = ir::makeLOr($expr0.expr, $expr1.expr);}
        ) expr1=exprNode
    | ( // max, min
        MAX {$expr = ir::makeMax($expr0.expr, $expr1.expr);}
        | MIN {$expr = ir::makeMin($expr0.expr, $expr1.expr);}
        ) LPAREN expr0=exprNode COMMA expr1=exprNode RPAREN
    | LPAREN exprNode RPAREN {$expr = $exprNode.expr;} // 放在最后以免破坏其他结构
    | INTRINSIC LPAREN DQUOTE  DQUOTE RPAREN; // Intrinsic: TODO 抓出中间结构

indices_expr returns [std::vector<ir::Expr> exprs]
    : exprNode (COMMA exprNode)* {
        std::cout << "indices_expr" << std::endl;
    };

load
    : Var LBRACK indices_expr RBRACK {
        std::cout << "load" << std::endl;
    };

var_params returns [std::vector <std::string> vec]
    : var=Var  {
        $vec.clear();
        $vec.push_back(std::string($var.text));
    } (COMMA var1=Var {$vec.push_back(std::string($var1.text));})*;

// PARAMETERS
shape returns [std::vector <ir::Expr> vec]
    : LBRACK exprNode {
        $vec.clear();
        $vec.emplace_back($exprNode.expr);
    } (COMMA newExprNode=exprNode {
        $vec.emplace_back($newExprNode.expr);
    }
    )* RBRACK;

// ---------- ---------- CONST ---------- ----------
intConst returns [ir::Expr expr]
    : Integer {
        const char *s = $Integer.text.c_str();
        $expr = ir::makeIntConst(std::atoll(s));
    };

floatConst returns [ir::Expr expr]
    : Float {
        auto s = $Float.text.c_str();
        $expr = ir::makeFloatConst(std::atof(s));
    };

boolConst returns [ir::Expr expr]
    @init {bool val;}
    : (
        TRUE {val = true;}
        |FALSE {val = false;}
    ) {$expr = ir::makeBoolConst(val);};