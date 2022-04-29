parser grammar pb_parser;

options {
    tokenVocab = pb_lexer;
}

@parser::postinclude {
    #include <expr.h>
}

func returns [std::vector<std::string> args, std::vector<Expr> values, Expr cond]
    : '{' varList '->' exprList (':' expr
      {
        $cond = $expr.node;
      }
        )? '}'
      {
        $args = $varList.vars;
        $values = $exprList.nodes;
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
        $node = makeVar($Id.text);
      }
    | '(' expr ')'
      {
        $node = $expr.node;
      }
    | expr0=expr '+' expr1=expr
      {
        $node = makeAdd($expr0.node, $expr1.node);
      }
    | expr0=expr '-' expr1=expr
      {
        $node = makeSub($expr0.node, $expr1.node);
      }
    | expr0=expr '*' expr1=expr
      {
        $node = makeMul($expr0.node, $expr1.node);
      }
    | FLOOR '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeFloorDiv($expr0.node, $expr1.node);
      }
    | CEIL '(' expr0=expr '/' expr1=expr ')'
      {
        $node = makeCeilDiv($expr0.node, $expr1.node);
      }
    | expr0=expr ('%' | MOD) expr1=expr
      {
        $node = makeMod($expr0.node, $expr1.node);
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
