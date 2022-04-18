lexer grammar MDLexer;

WhiteSpaces: [ \t\r\n]+ -> skip;

IF:         'if';
ELSE:       'else';
DO:         'do';
WHILE:      'while';
FOR:        'for';
RETURN:     'return';
BREAK:      'break';
CONTINUE:   'continue';

INT:        'int';

Integer:    [0-9]+;
Identifier: [a-zA-Z_][a-zA-Z0-9_]*;

ASSIGN:     '=';
PLUS:       '+';
MINUS:      '-';
STAR:       '*';
SLASH:      '/';
PERCENT:    '%';
NOT:        '!';
TILDE:      '~';
AND:        '&';
HAT:        '^';
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
LBRACK:     '{';
RBRACK:     '}';
COMMA:      ',';

