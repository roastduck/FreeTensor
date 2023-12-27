parser grammar pb_parser;

options {
    tokenVocab = pb_lexer;
}

@parser::postinclude {
    #include <expr.h>
    #include <debug.h>
    #include <serialize/load_ast.h>
    #include <serialize/mangle.h>
    #include <math/parse_pb_expr.h>
}

func returns [PBFuncAST ast]
    : (extList=varList '->')? '{' (simpleFunc
      {
        $ast = {$simpleFunc.ast};
      }
        (';' simpleFunc1=simpleFunc
      {
        $ast.emplace_back($simpleFunc.ast);
      }
        )*)? '}'
    ;

simpleFunc returns [SimplePBFuncAST ast]
    : varList '->' exprList (':' boolExpr
      {
        $ast.cond_ = $boolExpr.node;
      }
        )?
      {
        $ast.args_ = $varList.vars;
        $ast.values_ = $exprList.nodes;
      }
    ;

varList returns [std::vector<std::string> vars]
    : '[' (Id
      {
        $vars.emplace_back($Id.text);
      }
        (',' Id1=Id
      {
        $vars.emplace_back($Id1.text);
      }
        )*)? ']'
    ;

exprList returns [std::vector<Expr> nodes]
    : '[' (expr
      {
        $nodes.emplace_back($expr.node);
      }
        (',' expr1=expr
      {
        $nodes.emplace_back($expr1.node);
      }
        )*)? ']'
    ;

expr returns [Expr node]
    : intConst
      {
        $node = $intConst.node;
      }
    | Id
      {
        if (auto pos = $Id.text.find("__ext__"); pos != std::string::npos) {
            auto expr = loadAST(unmangle($Id.text.substr(0, pos)));
            ASSERT(expr->isExpr());
            $node = expr.as<ExprNode>();
        } else {
            $node = makeVar($Id.text);
        }
      }
    | '(' expr ')'
      {
        $node = $expr.node;
      }
    | FLOOR '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeFloorDiv($expr0.node, $expr1.node);
      }
    | CEIL '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeCeilDiv($expr0.node, $expr1.node);
      }
    | MIN '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMin($expr0.node, $expr1.node);
      }
    | MAX '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMax($expr0.node, $expr1.node);
      }
    | intConst expr
      {
        $node = makeMul($intConst.node, $expr.node);
      }
    | expr0=expr
      {int ty;} (
        '*' {ty = 1;}
        | ('%' | MOD) {ty = 2;}
      )
      expr1=expr
      {
        switch (ty)
        {
          case 1: $node = makeMul($expr0.node, $expr1.node); break;
          case 2: $node = makeMod($expr0.node, $expr1.node); break;
        }
      }
    | '-' expr0=expr
      {
        $node = makeSub(makeIntConst(0), $expr0.node);
      }
    | expr0=expr
      {int ty;} (
        '+' {ty = 1;}
        | '-' {ty = 2;}
      )
      expr1=expr
      {
        switch (ty)
        {
          case 1: $node = makeAdd($expr0.node, $expr1.node); break;
          case 2: $node = makeSub($expr0.node, $expr1.node); break;
        }
      }
    ;

boolExpr returns [Expr node]
    : '(' boolExpr0=boolExpr ')'
      {
        $node = $boolExpr0.node;
      }
    | expr0=expr
      {
        std::function<Expr(Expr, Expr)> make;
        $node = makeBoolConst(true);
        Expr last = $expr0.node;
      }
      (('<=' { make = [](Expr a, Expr b) { return makeLE(a, b); }; }
       |'>=' { make = [](Expr a, Expr b) { return makeGE(a, b); }; }
       |'<' { make = [](Expr a, Expr b) { return makeLT(a, b); }; }
       |'>' { make = [](Expr a, Expr b) { return makeGT(a, b); }; }
       |'=' { make = [](Expr a, Expr b) { return makeEQ(a, b); }; }
       |'!=' { make = [](Expr a, Expr b) { return makeNE(a, b); }; })
      expr1=expr { $node = makeLAnd($node, make(last, $expr1.node)); last = $expr1.node; })+
    | boolExpr0=boolExpr AND boolExpr1=boolExpr
      {
        $node = makeLAnd($boolExpr0.node, $boolExpr1.node);
      }
    | boolExpr0=boolExpr OR boolExpr1=boolExpr
      {
        $node = makeLOr($boolExpr0.node, $boolExpr1.node);
      }
    | NOT boolExpr
      {
        $node = makeLNot($boolExpr.node);
      }
    ;

intConst returns [Expr node]
    : Integer
      {
        $node = makeIntConst(std::stoll($Integer.text));
      }
    ;
