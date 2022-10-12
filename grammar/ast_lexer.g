lexer grammar ast_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;
Comment:    '/*' .*? '*/' -> skip;

IF:         'if';
ELSE:       'else';
FOR:        'for';
IN:         'in';
ASSERT_TOKEN:     'assert';
ASSUME:     'assume';
FUNC:       'func';

// empty
ANY:        'Any';

// VarDef
VIEW_OF:    '@!view_of';
PINNED:     '@!pinned';
ALLOC:      '@!alloc';
FREE:       '@!free';

// ReduceTo
ATOMIC:     '@!atomic';
PLUSEQ:     '+=';
SUBEQ:      '-=';
STAREQ:     '*=';
MINEQ:      '@!min=';
MAXEQ:      '@!max=';
ANDEQ:      '&&=';
OREQ:       '||=';

// For
NO_DEPS:    '@!no_deps';
PARALLEL:   '@!parallel';
REDUCTION:  '@!reduction';
UNROLL:     '@!unroll';
VECTORIZE:  '@!vectorize';
PREFERLIBS: '@!prefer_libs';

TRUE:       'true';
FALSE:      'false';

// expr
EVAL:       '@!eval';
FLOOR:      '@!floor';
CEIL:       '@!ceil';
ROUNDTO0:   '@!towards0';
MAX:        '@!max';
MIN:        '@!min';
SQRT:       '@!sqrt';
EXP:        '@!exp';
SQUARE:     '@!square';
ABS:        '@!abs';
SIGMOID:    '@!sigmoid';
TANH:       '@!tanh';
INTRINSIC:  '@!intrinsic';
SIDE_EFFECT:    '@!side_effect';
CLOSURE:    '@!closure';


Integer:    ('+'|'-')? [0-9]+;
Float:      (
              ('+'|'-')? [0-9]* '.' [0-9]* (('E'|'e') ('+'|'-')? [0-9]+)? |
              ('+'|'-')? ('0X'|'0x') [0-9a-fA-F]* '.' [0-9a-fA-F]* (('P'|'p') ('+'|'-')? [0-9]+)?
            );
String:     '"' ~["\r\n]+? '"';
SimpleVar:  [a-zA-Z_][a-zA-Z0-9_]*;
EscapedVar: '`' ~[`\r\n]+? '`';
AtVar:      '@' ~[ !\t\r\n]+;

DOT:        '.';
ASSIGN:     '=';
PLUS:       '+';
MINUS:      '-';
STAR:       '*';
SLASH:      '/';
PERCENT:    '%';
PERCENTPERCENT:    '%%';
NOT:        '!';
AND:        '&';
OR:         '|';
EQ:         '==';
NE:         '!=';
LT:         '<';
GT:         '>';
LE:         '<=';
GE:         '>=';
LAND:       '&&';
LOR:        '||';
COLON:      ':';
QUESTION:   '?';
LPAREN:     '(';
RPAREN:     ')';
LBRACK:     '[';
RBRACK:     ']';
LBRACE:     '{';
RBRACE:     '}';
COMMA:      ',';
RARROW:     '->';

BEGIN_META:   '#!' -> pushMode(INSIDE_METADATA);

mode INSIDE_METADATA;

WhiteSpacesMeta:    [ \t\r]+ -> skip;

INTEGER_META:       [0-9]+;
LABEL_META:         [a-zA-Z0-9_.\-]+;
ID_META:            '#' [0-9]+;
ANON_META:          '#<anon>';
LARROW_META:        '<~';
TRANSFORM_OP:       '$';
LBRACE_META:        '{';
RBRACE_META:        '}';

END_META:   '\n' -> popMode;
