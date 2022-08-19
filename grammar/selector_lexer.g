lexer grammar selector_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;

And: '&';
Or: '|';
Label: [a-zA-Z0-9_\-]+;
TransformOp: '$' [a-zA-Z0-9_\-.]+;
Id: '#' [0-9]*;

NodeTypeStmtSeq: '<StmtSeq>';
NodeTypeVarDef: '<VarDef>';
NodeTypeFor: '<For>';
NodeTypeIf: '<If>';
NodeTypeAssert: '<Assert>';
NodeTypeAssume: '<Assume>';

ChildArrow: '<-';
DescendantArrow: '<<-';
CallerArrow: '<~';

LeftBracket: '{';
RightBracket: '}';
Comma: ',';