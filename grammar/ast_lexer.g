lexer grammar ast_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;

IF:         'if';
ELSE:       'else';
DO:         'do';
WHILE:      'while';
FOR:        'for';
RETURN:     'return';
BREAK:      'break';
CONTINUE:   'continue';
FUNC:       'func';

// empty
ANY:        'Any';

// atype
IN:         'in';
OUT:        'out';
INOUT:      'inout';
CACHE:      'cache';

// mtype
BYVALUE:    'ByValue';
CPU:        'CPU';
GPUGlobal:  'GPUGlobal';
GPUShared:  'GPUShared';
GPULocal:   'GPULocal';
GPUWarp:    'GPUWarp';

// dtype
F32:        'f32';
F64:        'f64';
I32:        'i32';
BOOL:       'bool';
CUSTOM:     'custom';
VOID:       'void';

// VarDef
SIZELIM:    'size_lim';
PINNED:     '[pinned]';

// ReduceTo
ATOMIC:     '@atomic';
PLUSEQ:     '+=';
STAREQ:     '*=';
MINEQ:      'min=';
MAXEQ:      'max=';

// For
NO_DEPS:    '@no_deps';
PARALLEL:   '@parallel';
REDUCTION:  '@reduction';
UNROLL:     '@unroll';
VECTORIZE:  '@vectorize';
PREFERLIBS: '@prefer_libs';
// For parallel
OPENMP:     'openmp';
CUDASTREAM: 'cudastream';
SCOPE_BLOCK:'blockIdx';
SCOPE_THREAD:'threadIdx';
DOTX:       '.x';
DOTY:       '.y';
DOTZ:       '.z';

TRUE:       'true';
FALSE:      'false';

// expr
FLOOR:      'floor';
CEIL:       'ceil';
ROUNDTO0:   'towards0';
MAX:        'max';
MIN:        'min';
SQRT:       'sqrt';
EXP:        'exp';
SQUARE:     '^2';
ABS:        'abs';
SIGMOID:    'sigmoid';
TANH:       'tanh';
INTRINSIC:  'intrinsic';


Integer:    ('+'|'-')? [0-9]+;
Float:      ('+'|'-')? Integer '.' [0-9]* (('E'|'e') Integer)?;
SimpleVar:  [a-zA-Z_][a-zA-Z0-9_]*;
EscapedVar: '`' ~[`\r\n]+ '`';

DOT:        '.';
ASSIGN:     '=';
PLUS:       '+';
MINUS:      '-';
STAR:       '*';
SLASH:      '/';
PERCENT:    '%';
PERCENTPERCENT:    '%%';
NOT:        '!';
TILDE:      '~';
AND:        '&';
OR:         '|';
SL:         '<<';
SR:         '>>';
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
SEMICOLON:  ';';
LPAREN:     '(';
RPAREN:     ')';
LBRACK:     '[';
RBRACK:     ']';
LBRACE:     '{';
RBRACE:     '}';
COMMA:      ',';
DQUOTE:     '"';
RARROW:     '->';

