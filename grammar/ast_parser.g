parser grammar ast_parser;

options {
    tokenVocab = ast_lexer;
}

@parser::postinclude {
    #include <iostream>
    #include <string>
    #include <vector>

    #include <container_utils.h>
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
    : AtVar
      {
        $type = parseMType(slice($AtVar.text, 1));
      }
    ;

atype returns [AccessType type]
    : AtVar
      {
        $type = parseAType(slice($AtVar.text, 1));
      }
    ;

dtype returns [DataType type]
    : SimpleVar
      {
        $type = parseDType($SimpleVar.text);
      }
    ;

parallelScope returns [ParallelScope type]
    : AtVar
      {
        $type = parseParallelScope(slice($AtVar.text, 1));
      }
    ;

// TODO: closure?
func returns [Func node]
    @init {
        std::vector<std::pair<std::string, DataType>> ret;
    }
    : FUNC name=var '(' params ')'
        (RARROW retVals { ret = $retVals.vec; })?
        '{' stmts '}'
      {
        std::vector<std::pair<std::string, DataType>> returns;
        Stmt body;
        $node = makeFunc($name.name, $params.vec, std::move(ret), $stmts.node, {});
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
    : stmtWithoutID
      {
        $node = $stmtWithoutID.node;
      }
    | var ':' stmtWithoutID
      {
        $node = $stmtWithoutID.node;
        $node->setId($var.name);
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
    | for
      {
        $node = $for.node;
      }
    | if
      {
        $node = $if.node;
      }
    | assertNode
      {
        $node = $assertNode.node;
      }
    | assume
      {
        $node = $assume.node;
      }
    | expr
      {
        $node = makeEval(ID(), $expr.node);
      }
    | '{' '}'
      {
        $node = makeStmtSeq(ID(), {});
      }
    | '{' stmts '}'
      {
        $node = $stmts.node;
      }
    ;

store returns [Stmt node]
    : var indices '=' expr {
        $node = makeStore(ID(), $var.name, $indices.exprs, $expr.node);
    };

reduceTo returns [Stmt node]
    @init {
        bool atomic = false;
        ReduceOp op;
    }
    : (ATOMIC {atomic = true;})?
      var indices
        (PLUSEQ {op = ReduceOp::Add;}
        |STAREQ {op = ReduceOp::Mul;}
        |MINEQ {op = ReduceOp::Min;}
        |MAXEQ {op = ReduceOp::Max;})
        expr {
            $node = makeReduceTo(ID(), $var.name, $indices.exprs, op, $expr.node, atomic);
        }
    ;

load returns [Expr node]
    : var indices
      {
        $node = makeLoad($var.name, $indices.exprs);
      }
    ;

varDef returns [Stmt node]
    @init {
        Expr sizeLim = nullptr;
        bool pinned = false;
    }
    : atype mtype var ':' dtype shape
        (SIZELIM '=' expr {sizeLim = $expr.node;})?
        (PINNED {pinned = true;})?
        '{' stmts '}'
      {
        Ref<Tensor> t = makeTensor($shape.vec, $dtype.type);
        Ref<Buffer> b = makeBuffer(std::move(t), $atype.type, $mtype.type);
        Expr sizeLim = nullptr;
        Stmt body = $stmts.node;
        $node = makeVarDef(ID(), $var.name, std::move(b), std::move(sizeLim), body, pinned);
      }
    ;

// TODO: reduction
forProperty returns [Ref<ForProperty> property]
    : /* empty */
      {
        $property = Ref<ForProperty>::make();
      }
    | prev=forProperty NO_DEPS '=' var { std::vector<std::string> noDeps = {$var.name}; }
        (',' var2=var { noDeps.emplace_back($var2.name); })*
      {
        $property = $prev.property->withNoDeps(std::move(noDeps));
      }
    | prev=forProperty PREFERLIBS
      {
        $property = $prev.property->withPreferLibs();
      }
    | prev=forProperty UNROLL
      {
        $property = $prev.property->withUnroll();
      }
    | prev=forProperty VECTORIZE
      {
        $property = $prev.property->withVectorize();
      }
    | prev=forProperty PARALLEL '=' parallelScope
      {
        $property = $prev.property->withParallel($parallelScope.type);
      }
    ;

for returns [Stmt node]
    : forProperty
        FOR var IN begin=expr ':' end=expr ':' step=expr ':' len=expr
        '{' stmts '}'
      {
          $node = makeFor(ID(), $var.name, $begin.node, $end.node, $step.node, $len.node,
                          $forProperty.property, $stmts.node);
      }
    ;

if returns [Stmt node]
    : IF '(' cond=expr ')' '{' thenCase=stmts '}'
      {
        $node = makeIf(ID(), $cond.node, $thenCase.node);
      }
    | IF '(' cond=expr ')' '{' thenCase=stmts '}' ELSE '{' elseCase=stmts '}'
      {
        $node = makeIf(ID(), $cond.node, $thenCase.node, $elseCase.node);
      }
    ;

assertNode returns [Stmt node]
    : ASSERT_TOKEN '(' cond=expr ')' '{' stmts '}'
      {
        $node = makeAssert(ID(), $cond.node, $stmts.node);
      }
    ;

assume returns [Stmt node]
    : ASSUME '(' cond=expr ')' '{' stmts '}'
      {
        $node = makeAssume(ID(), $cond.node, $stmts.node);
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
    | var
      {
        $node = makeVar($var.name);
      }
    | '(' expr0=expr QUESTION expr1=expr ':' expr2=expr ')'
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
    | MIN '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMin($expr0.node, $expr1.node);
      }
    | MAX '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMax($expr0.node, $expr1.node);
      }
    | '(' expr ')'
      {
        $node = $expr.node;
      }
    | INTRINSIC '(' String '->' dtype { std::vector<Expr> params; bool hasSideEffect = false; }
        (',' expr { params.emplace_back($expr.node); } )*
        (',' SIDE_EFFECT { hasSideEffect = true; } )?
        ')'
      {
        $node = makeIntrinsic(slice($String.text, 1, -1), std::move(params), $dtype.type, hasSideEffect);
      }
    ;


// -------------------- PARAMETERS --------------------
shape returns [std::vector<Expr> vec]
    : '[' expr
      {
        $vec.emplace_back($expr.node);
      }
      (',' newExpr=expr
      {
        $vec.emplace_back($newExpr.node);
      }
      )* ']'
    | '[' ']'
    ;

params returns [std::vector<std::string> vec]
    : /* empty */
    | var { $vec.emplace_back($var.name); }
        (',' var1=var { $vec.emplace_back($var1.name); })*
    ;

retVals returns [std::vector<std::pair<std::string, DataType>> vec]
    : var ':' dtype { $vec.emplace_back($var.name, $dtype.type); }
        (',' var1=var ':' dtype1=dtype { $vec.emplace_back($var1.name, $dtype1.type); })*
    ;

indices returns [std::vector<Expr> exprs]
    : '[' expr
      {
        $exprs.emplace_back($expr.node);
      }
      (',' newExpr=expr
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

// -------------------- IDENTIFIER --------------------
var returns [std::string name]
    : SimpleVar
      {
        $name = $SimpleVar.text;
      }
    | EscapedVar
      {
        $name = std::string(slice($EscapedVar.text, 1, -1));
      }
    ;
