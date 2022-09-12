lexer grammar selector_lexer;

WhiteSpaces: [ \t\n\r]+ -> skip;

Not: '!';
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
NodeTypeStore: '<Store>';
NodeTypeReduceTo: '<ReduceTo>';
NodeTypeEval: '<Eval>';

ChildArrow: '<-';
DescendantArrow: '<<-';
RootNode: '-|';
DirectCallerArrow: '<~';
CallerArrow: '<<~';
RootCall: '~|';

Star: '*';

LeftBracket: '{';
RightBracket: '}';
LeftParen: '(';
RightParen: ')';
Comma: ',';
