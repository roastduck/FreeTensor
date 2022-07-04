lexer grammar pb_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;

MIN:    'min';
MAX:    'max';
FLOOR:  'floor';
CEIL:   'ceil';
MOD:    'mod';
AND:    'and';
OR:     'or';
NOT:    'not';

Integer:    ('+'|'-')? [0-9]+;
Id:  [a-zA-Z_][a-zA-Z0-9_]*;

PLUS:       '+';
MINUS:      '-';
STAR:       '*';
SLASH:      '/';
PERCENT:    '%';
LT:         '<';
GT:         '>';
LE:         '<=';
GE:         '>=';
EQ:         '=';
NE:         '!=';
COLON:      ':';
SEMICOLON:  ';';
LPAREN:     '(';
RPAREN:     ')';
LBRACK:     '[';
RBRACK:     ']';
LBRACE:     '{';
RBRACE:     '}';
COMMA:      ',';
RARROW:     '->';
