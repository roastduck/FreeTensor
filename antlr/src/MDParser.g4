parser grammar MDParser;

options {
    tokenVocab = MDLexer;
}

@parser::postinclude {

#include <string>
#include <vector>

#include "ASTNode.h"

}

program returns [std::shared_ptr<ProgramNode> node]
        : funcs EOF
          {
            $node = $funcs.node;
          }
        ;

funcs   returns [std::shared_ptr<ProgramNode> node]
        : /* empty */
          {
            $node = ProgramNode::make({});
          }
        | part=funcs func
          {
            $node = $part.node;
            $node->funcs_.push_back($func.node);
          }
        | part=funcs funcDec
          {
            $node = $part.node;
            $node->funcs_.push_back($funcDec.node);
          }
        | part=funcs gVarDef
          {
            $node = $part.node;
            $node->funcs_.push_back($gVarDef.node);
          }
        ;

func    returns [std::shared_ptr<FunctionNode> node]
        : INT Identifier '(' args ')' '{' stmtSeq '}'
          {
            $node = FunctionNode::make(ExprType::Int, $Identifier.text, $args.nodes, $stmtSeq.node);
          }
        ;

funcDec returns [std::shared_ptr<FunctionNode> node]
        : INT Identifier '(' args ')' ';'
          {
            $node = FunctionNode::make(ExprType::Int, $Identifier.text, $args.nodes, nullptr);
          }
        ;

gVarDef returns [std::shared_ptr<GlobalVarDefNode> node]
        : INT Identifier ';'
          {
            $node = GlobalVarDefNode::make(ExprType::Int, $Identifier.text);
          }
        | INT Identifier '=' expr ';'
          {
            $node = GlobalVarDefNode::make(ExprType::Int, $Identifier.text, $expr.text);
          }
        ;

stmtSeq returns [std::shared_ptr<StmtSeqNode> node]
        : /* empty */
          {
            $node = StmtSeqNode::make({}, true);
          }
        | part=stmtSeq stmt
          {
            $node = $part.node;
            $node->stmts_.push_back($stmt.node);
          }
        ;

stmt    returns [std::shared_ptr<StmtNode> node]
        : expr ';'
          {
            $node = InvokeNode::make($expr.node);
          }
        | varDefs ';'
          {
            $node = StmtSeqNode::make($varDefs.nodes);
          }
        | IF '(' expr ')' stmt
          {
            $node = IfThenElseNode::make($expr.node, $stmt.node);
          }
        | IF '(' expr ')' thenCase=stmt ELSE elseCase=stmt
          {
            $node = IfThenElseNode::make($expr.node, $thenCase.node, $elseCase.node);
          }
        | WHILE '(' expr ')' stmt
          {
            $node = WhileNode::make($expr.node, $stmt.node);
          }
        | DO stmt WHILE '(' expr ')' ';'
          {
            $node = DoWhileNode::make($expr.node, $stmt.node);
          }
        | FOR '(' init=mayExpr ';' cond=mayExpr ';' incr=mayExpr ')' stmt
          {
            std::shared_ptr<StmtNode> init, incr;
            std::shared_ptr<ExprNode> cond;
            if ($init.node) init = InvokeNode::make($init.node); else init = StmtSeqNode::make({});
            if ($cond.node) cond = $cond.node; else cond = IntegerNode::make(1);
            if ($incr.node) incr = InvokeNode::make($incr.node); else incr = StmtSeqNode::make({});
            $node = ForNode::make(init, cond, incr, $stmt.node);
          }
        | FOR '(' varDefs ';' cond=mayExpr ';' incr=mayExpr ')' stmt
          {
            std::shared_ptr<StmtNode> incr;
            std::shared_ptr<ExprNode> cond;
            if ($cond.node) cond = $cond.node; else cond = IntegerNode::make(1);
            if ($incr.node) incr = InvokeNode::make($incr.node); else incr = StmtSeqNode::make({});
            $node = ForNode::make(StmtSeqNode::make($varDefs.nodes), cond, incr, $stmt.node);
            $node = StmtSeqNode::make({$node}, true);
          }
        | RETURN expr ';'
          {
            $node = ReturnNode::make($expr.node);
          }
        | BREAK ';'
          {
            $node = BreakNode::make();
          }
        | CONTINUE ';'
          {
            $node = ContinueNode::make();
          }
        | ';'
          {
            $node = StmtSeqNode::make({});
          }
        | '{' stmtSeq '}'
          {
            $node = $stmtSeq.node;
          }
        ;

expr    returns [std::shared_ptr<ExprNode> node]
        : Integer
          {
            $node = IntegerNode::make(std::stoi($Integer.text));
          }
        | var
          {
            $node = $var.node;
          }
        | Identifier '(' exprs ')'
          {
            $node = CallNode::make(ExprType::Unknown, $Identifier.text, $exprs.nodes);
          }
        | '(' expr ')'
          {
            $node = $expr.node;
          }
        | '-' expr
          {
            $node = SubNode::make(IntegerNode::make(0), $expr.node);
          }
        | '!' expr
          {
            $node = LNotNode::make($expr.node);
          }
        | '~' expr
          {
            $node = BXorNode::make(IntegerNode::make(-1), $expr.node);
          }
        | lhs=expr op=('*' | '/' | '%') rhs=expr
          {
            if ($op.text == "*") {
                $node = MulNode::make($lhs.node, $rhs.node);
            } else if ($op.text == "/") {
                $node = DivNode::make($lhs.node, $rhs.node);
            } else if ($op.text == "%") {
                $node = ModNode::make($lhs.node, $rhs.node);
            }
          }
        | lhs=expr op=('+' | '-') rhs=expr
          {
            if ($op.text == "+") {
                $node = AddNode::make($lhs.node, $rhs.node);
            } else {
                $node = SubNode::make($lhs.node, $rhs.node);
            }
          }
        | lhs=expr op=('<<' | '>>') rhs=expr
          {
            if ($op.text == "<<") {
                $node = SLLNode::make($lhs.node, $rhs.node);
            } else if ($op.text == ">>") {
                $node = SRANode::make($lhs.node, $rhs.node);
            }
          }
        | lhs=expr op=('<' | '>' | '<=' | '>=') rhs=expr
          {
            if ($op.text == "<") {
                $node = LTNode::make($lhs.node, $rhs.node);
            } else if ($op.text == ">") {
                $node = GTNode::make($lhs.node, $rhs.node);
            } else if ($op.text == "<=") {
                $node = LENode::make($lhs.node, $rhs.node);
            } else if ($op.text == ">=") {
                $node = GENode::make($lhs.node, $rhs.node);
            }
          }
        | lhs=expr op=('==' | '!=') rhs=expr
          {
            if ($op.text == "==") {
                $node = EQNode::make($lhs.node, $rhs.node);
            } else if ($op.text == "!=") {
                $node = NENode::make($lhs.node, $rhs.node);
            }
          }
        | lhs=expr '&' rhs=expr
          {
            $node = BAndNode::make($lhs.node, $rhs.node);
          }
        | lhs=expr '^' rhs=expr
          {
            $node = BOrNode::make($lhs.node, $rhs.node);
          }
        | lhs=expr '|' rhs=expr
          {
            $node = BXorNode::make($lhs.node, $rhs.node);
          }
        | lhs=expr '&&' rhs=expr
          {
            $node = LAndNode::make($lhs.node, $rhs.node);
          }
        | lhs=expr '||' rhs=expr
          {
            $node = LOrNode::make($lhs.node, $rhs.node);
          }
        | a=expr '?' b=expr ':' c=expr
          {
            $node = SelectNode::make($a.node, $b.node, $c.node);
          }
        | Identifier '=' expr
          {
            $node = AssignNode::make($Identifier.text, $expr.node);
          }
        ;

exprs   returns [std::vector<std::shared_ptr<ExprNode>> nodes]
        : /* empty */
        | expr
          {
            $nodes = {$expr.node};
          }
        | part=exprs ',' expr
          {
            $nodes = $part.nodes;
            $nodes.push_back($expr.node);
          }
        ;

mayExpr returns [std::shared_ptr<ExprNode> node]
        : /* empty */
        | expr
          {
            $node = $expr.node;
          }
        ;

var     returns [std::shared_ptr<VarNode> node]
        : Identifier
          {
            $node = VarNode::make(ExprType::Unknown, $Identifier.text);
          }
        ;

vars    returns [std::vector<std::shared_ptr<VarNode>> nodes]
        : /* empty */
        | var
          {
            $nodes = {$var.node};
          }
        | part=vars ',' var
          {
            $nodes = $part.nodes;
            $nodes.push_back($var.node);
          }
        ;

varDef  returns [std::shared_ptr<StmtNode> node]
        : Identifier
          {
            $node = VarDefNode::make(ExprType::Int, $Identifier.text);
          }
        | Identifier '=' expr
          {
            $node = StmtSeqNode::make({
                        VarDefNode::make(ExprType::Int, $Identifier.text),
                        InvokeNode::make(AssignNode::make($Identifier.text, $expr.node))});
          }
        ;

varDefs returns [std::vector<std::shared_ptr<StmtNode>> nodes]
        : INT varDef
          {
            $nodes = {$varDef.node};
          }
        | part=varDefs ',' varDef
          {
            $nodes = $part.nodes;
            $nodes.push_back($varDef.node);
          }
        ;

arg     returns [std::pair<ExprType, std::string> node]
        : INT Identifier
          {
            $node = std::make_pair(ExprType::Int, $Identifier.text);
          }
        ;

args    returns [std::vector<std::pair<ExprType, std::string>> nodes]
        : /* empty */
        | arg
          {
            $nodes = {$arg.node};
          }
        | part=args ',' arg
          {
            $nodes = $part.nodes;
            $nodes.push_back($arg.node);
          }
        ;

