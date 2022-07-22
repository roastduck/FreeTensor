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
    : (extList=varList '->')? '{' simpleFunc
      {
        $ast = {$simpleFunc.ast};
      }
        (';' simpleFunc1=simpleFunc
      {
        $ast.emplace_back($simpleFunc.ast);
      }
        )* '}'
    ;

simpleFunc returns [SimplePBFuncAST ast]
    : varList '->' exprList (':' expr
      {
        $ast.cond_ = $expr.node;
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
    | MIN '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMin($expr0.node, $expr1.node);
      }
    | MAX '(' expr0=expr ',' expr1=expr ')'
      {
        $node = makeMax($expr0.node, $expr1.node);
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
    | expr0=expr '=' expr1=expr
      {
        $node = makeEQ($expr0.node, $expr1.node);
      }
    | expr0=expr '!=' expr1=expr
      {
        $node = makeNE($expr0.node, $expr1.node);
      }
    | expr0=expr AND expr1=expr
      {
        $node = makeLAnd($expr0.node, $expr1.node);
      }
    | expr0=expr OR expr1=expr
      {
        $node = makeLOr($expr0.node, $expr1.node);
      }
    | NOT expr
      {
        $node = makeLNot($expr.node);
      }
    | '-' expr0=expr
      {
        $node = makeSub(makeIntConst(0), $expr0.node);
      }
    ;

intConst returns [Expr node]
    : Integer
      {
        $node = makeIntConst(std::stoll($Integer.text));
      }
    ;
