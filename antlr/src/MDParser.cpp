
// Generated from MDParser.g4 by ANTLR 4.7.1



#include "MDParser.h"



#include <string>
#include <vector>

#include "ASTNode.h"



using namespace antlrcpp;
using namespace antlr4;

MDParser::MDParser(TokenStream *input) : Parser(input) {
  _interpreter = new atn::ParserATNSimulator(this, _atn, _decisionToDFA, _sharedContextCache);
}

MDParser::~MDParser() {
  delete _interpreter;
}

std::string MDParser::getGrammarFileName() const {
  return "MDParser.g4";
}

const std::vector<std::string>& MDParser::getRuleNames() const {
  return _ruleNames;
}

dfa::Vocabulary& MDParser::getVocabulary() const {
  return _vocabulary;
}


//----------------- ProgramContext ------------------------------------------------------------------

MDParser::ProgramContext::ProgramContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::FuncsContext* MDParser::ProgramContext::funcs() {
  return getRuleContext<MDParser::FuncsContext>(0);
}

tree::TerminalNode* MDParser::ProgramContext::EOF() {
  return getToken(MDParser::EOF, 0);
}


size_t MDParser::ProgramContext::getRuleIndex() const {
  return MDParser::RuleProgram;
}


MDParser::ProgramContext* MDParser::program() {
  ProgramContext *_localctx = _tracker.createInstance<ProgramContext>(_ctx, getState());
  enterRule(_localctx, 0, MDParser::RuleProgram);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(32);
    dynamic_cast<ProgramContext *>(_localctx)->funcsContext = funcs(0);
    setState(33);
    match(MDParser::EOF);

                dynamic_cast<ProgramContext *>(_localctx)->node =  dynamic_cast<ProgramContext *>(_localctx)->funcsContext->node;
              
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncsContext ------------------------------------------------------------------

MDParser::FuncsContext::FuncsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::FuncContext* MDParser::FuncsContext::func() {
  return getRuleContext<MDParser::FuncContext>(0);
}

MDParser::FuncsContext* MDParser::FuncsContext::funcs() {
  return getRuleContext<MDParser::FuncsContext>(0);
}

MDParser::FuncDecContext* MDParser::FuncsContext::funcDec() {
  return getRuleContext<MDParser::FuncDecContext>(0);
}

MDParser::GVarDefContext* MDParser::FuncsContext::gVarDef() {
  return getRuleContext<MDParser::GVarDefContext>(0);
}


size_t MDParser::FuncsContext::getRuleIndex() const {
  return MDParser::RuleFuncs;
}



MDParser::FuncsContext* MDParser::funcs() {
   return funcs(0);
}

MDParser::FuncsContext* MDParser::funcs(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::FuncsContext *_localctx = _tracker.createInstance<FuncsContext>(_ctx, parentState);
  MDParser::FuncsContext *previousContext = _localctx;
  size_t startState = 2;
  enterRecursionRule(_localctx, 2, MDParser::RuleFuncs, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);

                dynamic_cast<FuncsContext *>(_localctx)->node =  ProgramNode::make({});
              
    _ctx->stop = _input->LT(-1);
    setState(53);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(51);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 0, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<FuncsContext>(parentContext, parentState);
          _localctx->part = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleFuncs);
          setState(39);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(40);
          dynamic_cast<FuncsContext *>(_localctx)->funcContext = func();

                                dynamic_cast<FuncsContext *>(_localctx)->node =  dynamic_cast<FuncsContext *>(_localctx)->part->node;
                                _localctx->node->funcs_.push_back(dynamic_cast<FuncsContext *>(_localctx)->funcContext->node);
                              
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<FuncsContext>(parentContext, parentState);
          _localctx->part = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleFuncs);
          setState(43);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(44);
          dynamic_cast<FuncsContext *>(_localctx)->funcDecContext = funcDec();

                                dynamic_cast<FuncsContext *>(_localctx)->node =  dynamic_cast<FuncsContext *>(_localctx)->part->node;
                                _localctx->node->funcs_.push_back(dynamic_cast<FuncsContext *>(_localctx)->funcDecContext->node);
                              
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<FuncsContext>(parentContext, parentState);
          _localctx->part = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleFuncs);
          setState(47);

          if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
          setState(48);
          dynamic_cast<FuncsContext *>(_localctx)->gVarDefContext = gVarDef();

                                dynamic_cast<FuncsContext *>(_localctx)->node =  dynamic_cast<FuncsContext *>(_localctx)->part->node;
                                _localctx->node->funcs_.push_back(dynamic_cast<FuncsContext *>(_localctx)->gVarDefContext->node);
                              
          break;
        }

        } 
      }
      setState(55);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 1, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- FuncContext ------------------------------------------------------------------

MDParser::FuncContext::FuncContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::FuncContext::INT() {
  return getToken(MDParser::INT, 0);
}

tree::TerminalNode* MDParser::FuncContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}

MDParser::ArgsContext* MDParser::FuncContext::args() {
  return getRuleContext<MDParser::ArgsContext>(0);
}

MDParser::StmtSeqContext* MDParser::FuncContext::stmtSeq() {
  return getRuleContext<MDParser::StmtSeqContext>(0);
}


size_t MDParser::FuncContext::getRuleIndex() const {
  return MDParser::RuleFunc;
}


MDParser::FuncContext* MDParser::func() {
  FuncContext *_localctx = _tracker.createInstance<FuncContext>(_ctx, getState());
  enterRule(_localctx, 4, MDParser::RuleFunc);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(56);
    match(MDParser::INT);
    setState(57);
    dynamic_cast<FuncContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
    setState(58);
    match(MDParser::LPAREN);
    setState(59);
    dynamic_cast<FuncContext *>(_localctx)->argsContext = args(0);
    setState(60);
    match(MDParser::RPAREN);
    setState(61);
    match(MDParser::LBRACK);
    setState(62);
    dynamic_cast<FuncContext *>(_localctx)->stmtSeqContext = stmtSeq(0);
    setState(63);
    match(MDParser::RBRACK);

                dynamic_cast<FuncContext *>(_localctx)->node =  FunctionNode::make(ExprType::Int, (dynamic_cast<FuncContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<FuncContext *>(_localctx)->identifierToken->getText() : ""), dynamic_cast<FuncContext *>(_localctx)->argsContext->nodes, dynamic_cast<FuncContext *>(_localctx)->stmtSeqContext->node);
              
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- FuncDecContext ------------------------------------------------------------------

MDParser::FuncDecContext::FuncDecContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::FuncDecContext::INT() {
  return getToken(MDParser::INT, 0);
}

tree::TerminalNode* MDParser::FuncDecContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}

MDParser::ArgsContext* MDParser::FuncDecContext::args() {
  return getRuleContext<MDParser::ArgsContext>(0);
}


size_t MDParser::FuncDecContext::getRuleIndex() const {
  return MDParser::RuleFuncDec;
}


MDParser::FuncDecContext* MDParser::funcDec() {
  FuncDecContext *_localctx = _tracker.createInstance<FuncDecContext>(_ctx, getState());
  enterRule(_localctx, 6, MDParser::RuleFuncDec);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(66);
    match(MDParser::INT);
    setState(67);
    dynamic_cast<FuncDecContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
    setState(68);
    match(MDParser::LPAREN);
    setState(69);
    dynamic_cast<FuncDecContext *>(_localctx)->argsContext = args(0);
    setState(70);
    match(MDParser::RPAREN);
    setState(71);
    match(MDParser::SEMICOLON);

                dynamic_cast<FuncDecContext *>(_localctx)->node =  FunctionNode::make(ExprType::Int, (dynamic_cast<FuncDecContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<FuncDecContext *>(_localctx)->identifierToken->getText() : ""), dynamic_cast<FuncDecContext *>(_localctx)->argsContext->nodes, nullptr);
              
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- GVarDefContext ------------------------------------------------------------------

MDParser::GVarDefContext::GVarDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::GVarDefContext::INT() {
  return getToken(MDParser::INT, 0);
}

tree::TerminalNode* MDParser::GVarDefContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}

MDParser::ExprContext* MDParser::GVarDefContext::expr() {
  return getRuleContext<MDParser::ExprContext>(0);
}


size_t MDParser::GVarDefContext::getRuleIndex() const {
  return MDParser::RuleGVarDef;
}


MDParser::GVarDefContext* MDParser::gVarDef() {
  GVarDefContext *_localctx = _tracker.createInstance<GVarDefContext>(_ctx, getState());
  enterRule(_localctx, 8, MDParser::RuleGVarDef);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(85);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 2, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(74);
      match(MDParser::INT);
      setState(75);
      dynamic_cast<GVarDefContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
      setState(76);
      match(MDParser::SEMICOLON);

                  dynamic_cast<GVarDefContext *>(_localctx)->node =  GlobalVarDefNode::make(ExprType::Int, (dynamic_cast<GVarDefContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<GVarDefContext *>(_localctx)->identifierToken->getText() : ""));
                
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(78);
      match(MDParser::INT);
      setState(79);
      dynamic_cast<GVarDefContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
      setState(80);
      match(MDParser::ASSIGN);
      setState(81);
      dynamic_cast<GVarDefContext *>(_localctx)->exprContext = expr(0);
      setState(82);
      match(MDParser::SEMICOLON);

                  dynamic_cast<GVarDefContext *>(_localctx)->node =  GlobalVarDefNode::make(ExprType::Int, (dynamic_cast<GVarDefContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<GVarDefContext *>(_localctx)->identifierToken->getText() : ""), (dynamic_cast<GVarDefContext *>(_localctx)->exprContext != nullptr ? _input->getText(dynamic_cast<GVarDefContext *>(_localctx)->exprContext->start, dynamic_cast<GVarDefContext *>(_localctx)->exprContext->stop) : nullptr));
                
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- StmtSeqContext ------------------------------------------------------------------

MDParser::StmtSeqContext::StmtSeqContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::StmtContext* MDParser::StmtSeqContext::stmt() {
  return getRuleContext<MDParser::StmtContext>(0);
}

MDParser::StmtSeqContext* MDParser::StmtSeqContext::stmtSeq() {
  return getRuleContext<MDParser::StmtSeqContext>(0);
}


size_t MDParser::StmtSeqContext::getRuleIndex() const {
  return MDParser::RuleStmtSeq;
}



MDParser::StmtSeqContext* MDParser::stmtSeq() {
   return stmtSeq(0);
}

MDParser::StmtSeqContext* MDParser::stmtSeq(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::StmtSeqContext *_localctx = _tracker.createInstance<StmtSeqContext>(_ctx, parentState);
  MDParser::StmtSeqContext *previousContext = _localctx;
  size_t startState = 10;
  enterRecursionRule(_localctx, 10, MDParser::RuleStmtSeq, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);

                dynamic_cast<StmtSeqContext *>(_localctx)->node =  StmtSeqNode::make({}, true);
              
    _ctx->stop = _input->LT(-1);
    setState(96);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<StmtSeqContext>(parentContext, parentState);
        _localctx->part = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleStmtSeq);
        setState(90);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(91);
        dynamic_cast<StmtSeqContext *>(_localctx)->stmtContext = stmt();

                              dynamic_cast<StmtSeqContext *>(_localctx)->node =  dynamic_cast<StmtSeqContext *>(_localctx)->part->node;
                              _localctx->node->stmts_.push_back(dynamic_cast<StmtSeqContext *>(_localctx)->stmtContext->node);
                             
      }
      setState(98);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 3, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- StmtContext ------------------------------------------------------------------

MDParser::StmtContext::StmtContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::ExprContext* MDParser::StmtContext::expr() {
  return getRuleContext<MDParser::ExprContext>(0);
}

MDParser::VarDefsContext* MDParser::StmtContext::varDefs() {
  return getRuleContext<MDParser::VarDefsContext>(0);
}

tree::TerminalNode* MDParser::StmtContext::IF() {
  return getToken(MDParser::IF, 0);
}

std::vector<MDParser::StmtContext *> MDParser::StmtContext::stmt() {
  return getRuleContexts<MDParser::StmtContext>();
}

MDParser::StmtContext* MDParser::StmtContext::stmt(size_t i) {
  return getRuleContext<MDParser::StmtContext>(i);
}

tree::TerminalNode* MDParser::StmtContext::ELSE() {
  return getToken(MDParser::ELSE, 0);
}

tree::TerminalNode* MDParser::StmtContext::WHILE() {
  return getToken(MDParser::WHILE, 0);
}

tree::TerminalNode* MDParser::StmtContext::DO() {
  return getToken(MDParser::DO, 0);
}

tree::TerminalNode* MDParser::StmtContext::FOR() {
  return getToken(MDParser::FOR, 0);
}

std::vector<MDParser::MayExprContext *> MDParser::StmtContext::mayExpr() {
  return getRuleContexts<MDParser::MayExprContext>();
}

MDParser::MayExprContext* MDParser::StmtContext::mayExpr(size_t i) {
  return getRuleContext<MDParser::MayExprContext>(i);
}

tree::TerminalNode* MDParser::StmtContext::RETURN() {
  return getToken(MDParser::RETURN, 0);
}

tree::TerminalNode* MDParser::StmtContext::BREAK() {
  return getToken(MDParser::BREAK, 0);
}

tree::TerminalNode* MDParser::StmtContext::CONTINUE() {
  return getToken(MDParser::CONTINUE, 0);
}

MDParser::StmtSeqContext* MDParser::StmtContext::stmtSeq() {
  return getRuleContext<MDParser::StmtSeqContext>(0);
}


size_t MDParser::StmtContext::getRuleIndex() const {
  return MDParser::RuleStmt;
}


MDParser::StmtContext* MDParser::stmt() {
  StmtContext *_localctx = _tracker.createInstance<StmtContext>(_ctx, getState());
  enterRule(_localctx, 12, MDParser::RuleStmt);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(179);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 4, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(99);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(100);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  InvokeNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node);
                
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(103);
      dynamic_cast<StmtContext *>(_localctx)->varDefsContext = varDefs(0);
      setState(104);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  StmtSeqNode::make(dynamic_cast<StmtContext *>(_localctx)->varDefsContext->nodes);
                
      break;
    }

    case 3: {
      enterOuterAlt(_localctx, 3);
      setState(107);
      match(MDParser::IF);
      setState(108);
      match(MDParser::LPAREN);
      setState(109);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(110);
      match(MDParser::RPAREN);
      setState(111);
      dynamic_cast<StmtContext *>(_localctx)->stmtContext = stmt();

                  dynamic_cast<StmtContext *>(_localctx)->node =  IfThenElseNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node, dynamic_cast<StmtContext *>(_localctx)->stmtContext->node);
                
      break;
    }

    case 4: {
      enterOuterAlt(_localctx, 4);
      setState(114);
      match(MDParser::IF);
      setState(115);
      match(MDParser::LPAREN);
      setState(116);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(117);
      match(MDParser::RPAREN);
      setState(118);
      dynamic_cast<StmtContext *>(_localctx)->thenCase = stmt();
      setState(119);
      match(MDParser::ELSE);
      setState(120);
      dynamic_cast<StmtContext *>(_localctx)->elseCase = stmt();

                  dynamic_cast<StmtContext *>(_localctx)->node =  IfThenElseNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node, dynamic_cast<StmtContext *>(_localctx)->thenCase->node, dynamic_cast<StmtContext *>(_localctx)->elseCase->node);
                
      break;
    }

    case 5: {
      enterOuterAlt(_localctx, 5);
      setState(123);
      match(MDParser::WHILE);
      setState(124);
      match(MDParser::LPAREN);
      setState(125);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(126);
      match(MDParser::RPAREN);
      setState(127);
      dynamic_cast<StmtContext *>(_localctx)->stmtContext = stmt();

                  dynamic_cast<StmtContext *>(_localctx)->node =  WhileNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node, dynamic_cast<StmtContext *>(_localctx)->stmtContext->node);
                
      break;
    }

    case 6: {
      enterOuterAlt(_localctx, 6);
      setState(130);
      match(MDParser::DO);
      setState(131);
      dynamic_cast<StmtContext *>(_localctx)->stmtContext = stmt();
      setState(132);
      match(MDParser::WHILE);
      setState(133);
      match(MDParser::LPAREN);
      setState(134);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(135);
      match(MDParser::RPAREN);
      setState(136);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  DoWhileNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node, dynamic_cast<StmtContext *>(_localctx)->stmtContext->node);
                
      break;
    }

    case 7: {
      enterOuterAlt(_localctx, 7);
      setState(139);
      match(MDParser::FOR);
      setState(140);
      match(MDParser::LPAREN);
      setState(141);
      dynamic_cast<StmtContext *>(_localctx)->init = mayExpr();
      setState(142);
      match(MDParser::SEMICOLON);
      setState(143);
      dynamic_cast<StmtContext *>(_localctx)->cond = mayExpr();
      setState(144);
      match(MDParser::SEMICOLON);
      setState(145);
      dynamic_cast<StmtContext *>(_localctx)->incr = mayExpr();
      setState(146);
      match(MDParser::RPAREN);
      setState(147);
      dynamic_cast<StmtContext *>(_localctx)->stmtContext = stmt();

                  std::shared_ptr<StmtNode> init, incr;
                  std::shared_ptr<ExprNode> cond;
                  if (dynamic_cast<StmtContext *>(_localctx)->init->node) init = InvokeNode::make(dynamic_cast<StmtContext *>(_localctx)->init->node); else init = StmtSeqNode::make({});
                  if (dynamic_cast<StmtContext *>(_localctx)->cond->node) cond = dynamic_cast<StmtContext *>(_localctx)->cond->node; else cond = IntegerNode::make(1);
                  if (dynamic_cast<StmtContext *>(_localctx)->incr->node) incr = InvokeNode::make(dynamic_cast<StmtContext *>(_localctx)->incr->node); else incr = StmtSeqNode::make({});
                  dynamic_cast<StmtContext *>(_localctx)->node =  ForNode::make(init, cond, incr, dynamic_cast<StmtContext *>(_localctx)->stmtContext->node);
                
      break;
    }

    case 8: {
      enterOuterAlt(_localctx, 8);
      setState(150);
      match(MDParser::FOR);
      setState(151);
      match(MDParser::LPAREN);
      setState(152);
      dynamic_cast<StmtContext *>(_localctx)->varDefsContext = varDefs(0);
      setState(153);
      match(MDParser::SEMICOLON);
      setState(154);
      dynamic_cast<StmtContext *>(_localctx)->cond = mayExpr();
      setState(155);
      match(MDParser::SEMICOLON);
      setState(156);
      dynamic_cast<StmtContext *>(_localctx)->incr = mayExpr();
      setState(157);
      match(MDParser::RPAREN);
      setState(158);
      dynamic_cast<StmtContext *>(_localctx)->stmtContext = stmt();

                  std::shared_ptr<StmtNode> incr;
                  std::shared_ptr<ExprNode> cond;
                  if (dynamic_cast<StmtContext *>(_localctx)->cond->node) cond = dynamic_cast<StmtContext *>(_localctx)->cond->node; else cond = IntegerNode::make(1);
                  if (dynamic_cast<StmtContext *>(_localctx)->incr->node) incr = InvokeNode::make(dynamic_cast<StmtContext *>(_localctx)->incr->node); else incr = StmtSeqNode::make({});
                  dynamic_cast<StmtContext *>(_localctx)->node =  ForNode::make(StmtSeqNode::make(dynamic_cast<StmtContext *>(_localctx)->varDefsContext->nodes), cond, incr, dynamic_cast<StmtContext *>(_localctx)->stmtContext->node);
                  dynamic_cast<StmtContext *>(_localctx)->node =  StmtSeqNode::make({_localctx->node}, true);
                
      break;
    }

    case 9: {
      enterOuterAlt(_localctx, 9);
      setState(161);
      match(MDParser::RETURN);
      setState(162);
      dynamic_cast<StmtContext *>(_localctx)->exprContext = expr(0);
      setState(163);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  ReturnNode::make(dynamic_cast<StmtContext *>(_localctx)->exprContext->node);
                
      break;
    }

    case 10: {
      enterOuterAlt(_localctx, 10);
      setState(166);
      match(MDParser::BREAK);
      setState(167);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  BreakNode::make();
                
      break;
    }

    case 11: {
      enterOuterAlt(_localctx, 11);
      setState(169);
      match(MDParser::CONTINUE);
      setState(170);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  ContinueNode::make();
                
      break;
    }

    case 12: {
      enterOuterAlt(_localctx, 12);
      setState(172);
      match(MDParser::SEMICOLON);

                  dynamic_cast<StmtContext *>(_localctx)->node =  StmtSeqNode::make({});
                
      break;
    }

    case 13: {
      enterOuterAlt(_localctx, 13);
      setState(174);
      match(MDParser::LBRACK);
      setState(175);
      dynamic_cast<StmtContext *>(_localctx)->stmtSeqContext = stmtSeq(0);
      setState(176);
      match(MDParser::RBRACK);

                  dynamic_cast<StmtContext *>(_localctx)->node =  dynamic_cast<StmtContext *>(_localctx)->stmtSeqContext->node;
                
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ExprContext ------------------------------------------------------------------

MDParser::ExprContext::ExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::ExprContext::Integer() {
  return getToken(MDParser::Integer, 0);
}

MDParser::VarContext* MDParser::ExprContext::var() {
  return getRuleContext<MDParser::VarContext>(0);
}

tree::TerminalNode* MDParser::ExprContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}

MDParser::ExprsContext* MDParser::ExprContext::exprs() {
  return getRuleContext<MDParser::ExprsContext>(0);
}

std::vector<MDParser::ExprContext *> MDParser::ExprContext::expr() {
  return getRuleContexts<MDParser::ExprContext>();
}

MDParser::ExprContext* MDParser::ExprContext::expr(size_t i) {
  return getRuleContext<MDParser::ExprContext>(i);
}


size_t MDParser::ExprContext::getRuleIndex() const {
  return MDParser::RuleExpr;
}



MDParser::ExprContext* MDParser::expr() {
   return expr(0);
}

MDParser::ExprContext* MDParser::expr(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::ExprContext *_localctx = _tracker.createInstance<ExprContext>(_ctx, parentState);
  MDParser::ExprContext *previousContext = _localctx;
  size_t startState = 14;
  enterRecursionRule(_localctx, 14, MDParser::RuleExpr, precedence);

    size_t _la = 0;

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(215);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 5, _ctx)) {
    case 1: {
      setState(182);
      dynamic_cast<ExprContext *>(_localctx)->integerToken = match(MDParser::Integer);

                  dynamic_cast<ExprContext *>(_localctx)->node =  IntegerNode::make(std::stoi((dynamic_cast<ExprContext *>(_localctx)->integerToken != nullptr ? dynamic_cast<ExprContext *>(_localctx)->integerToken->getText() : "")));
                
      break;
    }

    case 2: {
      setState(184);
      dynamic_cast<ExprContext *>(_localctx)->varContext = var();

                  dynamic_cast<ExprContext *>(_localctx)->node =  dynamic_cast<ExprContext *>(_localctx)->varContext->node;
                
      break;
    }

    case 3: {
      setState(187);
      dynamic_cast<ExprContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
      setState(188);
      match(MDParser::LPAREN);
      setState(189);
      dynamic_cast<ExprContext *>(_localctx)->exprsContext = exprs(0);
      setState(190);
      match(MDParser::RPAREN);

                  dynamic_cast<ExprContext *>(_localctx)->node =  CallNode::make(ExprType::Unknown, (dynamic_cast<ExprContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<ExprContext *>(_localctx)->identifierToken->getText() : ""), dynamic_cast<ExprContext *>(_localctx)->exprsContext->nodes);
                
      break;
    }

    case 4: {
      setState(193);
      match(MDParser::LPAREN);
      setState(194);
      dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(0);
      setState(195);
      match(MDParser::RPAREN);

                  dynamic_cast<ExprContext *>(_localctx)->node =  dynamic_cast<ExprContext *>(_localctx)->exprContext->node;
                
      break;
    }

    case 5: {
      setState(198);
      match(MDParser::MINUS);
      setState(199);
      dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(15);

                  dynamic_cast<ExprContext *>(_localctx)->node =  SubNode::make(IntegerNode::make(0), dynamic_cast<ExprContext *>(_localctx)->exprContext->node);
                
      break;
    }

    case 6: {
      setState(202);
      match(MDParser::NOT);
      setState(203);
      dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(14);

                  dynamic_cast<ExprContext *>(_localctx)->node =  LNotNode::make(dynamic_cast<ExprContext *>(_localctx)->exprContext->node);
                
      break;
    }

    case 7: {
      setState(206);
      match(MDParser::TILDE);
      setState(207);
      dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(13);

                  dynamic_cast<ExprContext *>(_localctx)->node =  BXorNode::make(IntegerNode::make(-1), dynamic_cast<ExprContext *>(_localctx)->exprContext->node);
                
      break;
    }

    case 8: {
      setState(210);
      dynamic_cast<ExprContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
      setState(211);
      match(MDParser::ASSIGN);
      setState(212);
      dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(1);

                  dynamic_cast<ExprContext *>(_localctx)->node =  AssignNode::make((dynamic_cast<ExprContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<ExprContext *>(_localctx)->identifierToken->getText() : ""), dynamic_cast<ExprContext *>(_localctx)->exprContext->node);
                
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(276);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        setState(274);
        _errHandler->sync(this);
        switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 6, _ctx)) {
        case 1: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(217);

          if (!(precpred(_ctx, 12))) throw FailedPredicateException(this, "precpred(_ctx, 12)");
          setState(218);
          dynamic_cast<ExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & ((1ULL << MDParser::STAR)
            | (1ULL << MDParser::SLASH)
            | (1ULL << MDParser::PERCENT))) != 0))) {
            dynamic_cast<ExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(219);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(13);

                                if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "*") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  MulNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "/") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  DivNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "%") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  ModNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                }
                              
          break;
        }

        case 2: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(222);

          if (!(precpred(_ctx, 11))) throw FailedPredicateException(this, "precpred(_ctx, 11)");
          setState(223);
          dynamic_cast<ExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == MDParser::PLUS

          || _la == MDParser::MINUS)) {
            dynamic_cast<ExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(224);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(12);

                                if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "+") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  AddNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  SubNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                }
                              
          break;
        }

        case 3: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(227);

          if (!(precpred(_ctx, 10))) throw FailedPredicateException(this, "precpred(_ctx, 10)");
          setState(228);
          dynamic_cast<ExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == MDParser::SL

          || _la == MDParser::SR)) {
            dynamic_cast<ExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(229);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(11);

                                if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "<<") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  SLLNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == ">>") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  SRANode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                }
                              
          break;
        }

        case 4: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(232);

          if (!(precpred(_ctx, 9))) throw FailedPredicateException(this, "precpred(_ctx, 9)");
          setState(233);
          dynamic_cast<ExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!((((_la & ~ 0x3fULL) == 0) &&
            ((1ULL << _la) & ((1ULL << MDParser::LT)
            | (1ULL << MDParser::GT)
            | (1ULL << MDParser::LE)
            | (1ULL << MDParser::GE))) != 0))) {
            dynamic_cast<ExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(234);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(10);

                                if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "<") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  LTNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == ">") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  GTNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "<=") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  LENode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == ">=") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  GENode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                }
                              
          break;
        }

        case 5: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(237);

          if (!(precpred(_ctx, 8))) throw FailedPredicateException(this, "precpred(_ctx, 8)");
          setState(238);
          dynamic_cast<ExprContext *>(_localctx)->op = _input->LT(1);
          _la = _input->LA(1);
          if (!(_la == MDParser::EQ

          || _la == MDParser::NE)) {
            dynamic_cast<ExprContext *>(_localctx)->op = _errHandler->recoverInline(this);
          }
          else {
            _errHandler->reportMatch(this);
            consume();
          }
          setState(239);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(9);

                                if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "==") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  EQNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                } else if ((dynamic_cast<ExprContext *>(_localctx)->op != nullptr ? dynamic_cast<ExprContext *>(_localctx)->op->getText() : "") == "!=") {
                                    dynamic_cast<ExprContext *>(_localctx)->node =  NENode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                                }
                              
          break;
        }

        case 6: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(242);

          if (!(precpred(_ctx, 7))) throw FailedPredicateException(this, "precpred(_ctx, 7)");
          setState(243);
          match(MDParser::AND);
          setState(244);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(8);

                                dynamic_cast<ExprContext *>(_localctx)->node =  BAndNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                              
          break;
        }

        case 7: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(247);

          if (!(precpred(_ctx, 6))) throw FailedPredicateException(this, "precpred(_ctx, 6)");
          setState(248);
          match(MDParser::HAT);
          setState(249);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(7);

                                dynamic_cast<ExprContext *>(_localctx)->node =  BOrNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                              
          break;
        }

        case 8: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(252);

          if (!(precpred(_ctx, 5))) throw FailedPredicateException(this, "precpred(_ctx, 5)");
          setState(253);
          match(MDParser::OR);
          setState(254);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(6);

                                dynamic_cast<ExprContext *>(_localctx)->node =  BXorNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                              
          break;
        }

        case 9: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(257);

          if (!(precpred(_ctx, 4))) throw FailedPredicateException(this, "precpred(_ctx, 4)");
          setState(258);
          match(MDParser::LAND);
          setState(259);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(5);

                                dynamic_cast<ExprContext *>(_localctx)->node =  LAndNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                              
          break;
        }

        case 10: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->lhs = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(262);

          if (!(precpred(_ctx, 3))) throw FailedPredicateException(this, "precpred(_ctx, 3)");
          setState(263);
          match(MDParser::LOR);
          setState(264);
          dynamic_cast<ExprContext *>(_localctx)->rhs = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(4);

                                dynamic_cast<ExprContext *>(_localctx)->node =  LOrNode::make(dynamic_cast<ExprContext *>(_localctx)->lhs->node, dynamic_cast<ExprContext *>(_localctx)->rhs->node);
                              
          break;
        }

        case 11: {
          _localctx = _tracker.createInstance<ExprContext>(parentContext, parentState);
          _localctx->a = previousContext;
          pushNewRecursionContext(_localctx, startState, RuleExpr);
          setState(267);

          if (!(precpred(_ctx, 2))) throw FailedPredicateException(this, "precpred(_ctx, 2)");
          setState(268);
          match(MDParser::QUESTION);
          setState(269);
          dynamic_cast<ExprContext *>(_localctx)->b = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(0);
          setState(270);
          match(MDParser::COLON);
          setState(271);
          dynamic_cast<ExprContext *>(_localctx)->c = dynamic_cast<ExprContext *>(_localctx)->exprContext = expr(3);

                                dynamic_cast<ExprContext *>(_localctx)->node =  SelectNode::make(dynamic_cast<ExprContext *>(_localctx)->a->node, dynamic_cast<ExprContext *>(_localctx)->b->node, dynamic_cast<ExprContext *>(_localctx)->c->node);
                              
          break;
        }

        } 
      }
      setState(278);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 7, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ExprsContext ------------------------------------------------------------------

MDParser::ExprsContext::ExprsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::ExprContext* MDParser::ExprsContext::expr() {
  return getRuleContext<MDParser::ExprContext>(0);
}

MDParser::ExprsContext* MDParser::ExprsContext::exprs() {
  return getRuleContext<MDParser::ExprsContext>(0);
}


size_t MDParser::ExprsContext::getRuleIndex() const {
  return MDParser::RuleExprs;
}



MDParser::ExprsContext* MDParser::exprs() {
   return exprs(0);
}

MDParser::ExprsContext* MDParser::exprs(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::ExprsContext *_localctx = _tracker.createInstance<ExprsContext>(_ctx, parentState);
  MDParser::ExprsContext *previousContext = _localctx;
  size_t startState = 16;
  enterRecursionRule(_localctx, 16, MDParser::RuleExprs, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(283);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 8, _ctx)) {
    case 1: {
      break;
    }

    case 2: {
      setState(280);
      dynamic_cast<ExprsContext *>(_localctx)->exprContext = expr(0);

                  dynamic_cast<ExprsContext *>(_localctx)->nodes =  {dynamic_cast<ExprsContext *>(_localctx)->exprContext->node};
                
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(292);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ExprsContext>(parentContext, parentState);
        _localctx->part = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleExprs);
        setState(285);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(286);
        match(MDParser::COMMA);
        setState(287);
        dynamic_cast<ExprsContext *>(_localctx)->exprContext = expr(0);

                              dynamic_cast<ExprsContext *>(_localctx)->nodes =  dynamic_cast<ExprsContext *>(_localctx)->part->nodes;
                              _localctx->nodes.push_back(dynamic_cast<ExprsContext *>(_localctx)->exprContext->node);
                             
      }
      setState(294);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 9, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- MayExprContext ------------------------------------------------------------------

MDParser::MayExprContext::MayExprContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::ExprContext* MDParser::MayExprContext::expr() {
  return getRuleContext<MDParser::ExprContext>(0);
}


size_t MDParser::MayExprContext::getRuleIndex() const {
  return MDParser::RuleMayExpr;
}


MDParser::MayExprContext* MDParser::mayExpr() {
  MayExprContext *_localctx = _tracker.createInstance<MayExprContext>(_ctx, getState());
  enterRule(_localctx, 18, MDParser::RuleMayExpr);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(299);
    _errHandler->sync(this);
    switch (_input->LA(1)) {
      case MDParser::SEMICOLON:
      case MDParser::RPAREN: {
        enterOuterAlt(_localctx, 1);

        break;
      }

      case MDParser::Integer:
      case MDParser::Identifier:
      case MDParser::MINUS:
      case MDParser::NOT:
      case MDParser::TILDE:
      case MDParser::LPAREN: {
        enterOuterAlt(_localctx, 2);
        setState(296);
        dynamic_cast<MayExprContext *>(_localctx)->exprContext = expr(0);

                    dynamic_cast<MayExprContext *>(_localctx)->node =  dynamic_cast<MayExprContext *>(_localctx)->exprContext->node;
                  
        break;
      }

    default:
      throw NoViableAltException(this);
    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarContext ------------------------------------------------------------------

MDParser::VarContext::VarContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::VarContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}


size_t MDParser::VarContext::getRuleIndex() const {
  return MDParser::RuleVar;
}


MDParser::VarContext* MDParser::var() {
  VarContext *_localctx = _tracker.createInstance<VarContext>(_ctx, getState());
  enterRule(_localctx, 20, MDParser::RuleVar);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(301);
    dynamic_cast<VarContext *>(_localctx)->identifierToken = match(MDParser::Identifier);

                dynamic_cast<VarContext *>(_localctx)->node =  VarNode::make(ExprType::Unknown, (dynamic_cast<VarContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<VarContext *>(_localctx)->identifierToken->getText() : ""));
              
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarsContext ------------------------------------------------------------------

MDParser::VarsContext::VarsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::VarContext* MDParser::VarsContext::var() {
  return getRuleContext<MDParser::VarContext>(0);
}

MDParser::VarsContext* MDParser::VarsContext::vars() {
  return getRuleContext<MDParser::VarsContext>(0);
}


size_t MDParser::VarsContext::getRuleIndex() const {
  return MDParser::RuleVars;
}



MDParser::VarsContext* MDParser::vars() {
   return vars(0);
}

MDParser::VarsContext* MDParser::vars(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::VarsContext *_localctx = _tracker.createInstance<VarsContext>(_ctx, parentState);
  MDParser::VarsContext *previousContext = _localctx;
  size_t startState = 22;
  enterRecursionRule(_localctx, 22, MDParser::RuleVars, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(308);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 11, _ctx)) {
    case 1: {
      break;
    }

    case 2: {
      setState(305);
      dynamic_cast<VarsContext *>(_localctx)->varContext = var();

                  dynamic_cast<VarsContext *>(_localctx)->nodes =  {dynamic_cast<VarsContext *>(_localctx)->varContext->node};
                
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(317);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<VarsContext>(parentContext, parentState);
        _localctx->part = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleVars);
        setState(310);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(311);
        match(MDParser::COMMA);
        setState(312);
        dynamic_cast<VarsContext *>(_localctx)->varContext = var();

                              dynamic_cast<VarsContext *>(_localctx)->nodes =  dynamic_cast<VarsContext *>(_localctx)->part->nodes;
                              _localctx->nodes.push_back(dynamic_cast<VarsContext *>(_localctx)->varContext->node);
                             
      }
      setState(319);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 12, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- VarDefContext ------------------------------------------------------------------

MDParser::VarDefContext::VarDefContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::VarDefContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}

MDParser::ExprContext* MDParser::VarDefContext::expr() {
  return getRuleContext<MDParser::ExprContext>(0);
}


size_t MDParser::VarDefContext::getRuleIndex() const {
  return MDParser::RuleVarDef;
}


MDParser::VarDefContext* MDParser::varDef() {
  VarDefContext *_localctx = _tracker.createInstance<VarDefContext>(_ctx, getState());
  enterRule(_localctx, 24, MDParser::RuleVarDef);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    setState(327);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 13, _ctx)) {
    case 1: {
      enterOuterAlt(_localctx, 1);
      setState(320);
      dynamic_cast<VarDefContext *>(_localctx)->identifierToken = match(MDParser::Identifier);

                  dynamic_cast<VarDefContext *>(_localctx)->node =  VarDefNode::make(ExprType::Int, (dynamic_cast<VarDefContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<VarDefContext *>(_localctx)->identifierToken->getText() : ""));
                
      break;
    }

    case 2: {
      enterOuterAlt(_localctx, 2);
      setState(322);
      dynamic_cast<VarDefContext *>(_localctx)->identifierToken = match(MDParser::Identifier);
      setState(323);
      match(MDParser::ASSIGN);
      setState(324);
      dynamic_cast<VarDefContext *>(_localctx)->exprContext = expr(0);

                  dynamic_cast<VarDefContext *>(_localctx)->node =  StmtSeqNode::make({
                              VarDefNode::make(ExprType::Int, (dynamic_cast<VarDefContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<VarDefContext *>(_localctx)->identifierToken->getText() : "")),
                              InvokeNode::make(AssignNode::make((dynamic_cast<VarDefContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<VarDefContext *>(_localctx)->identifierToken->getText() : ""), dynamic_cast<VarDefContext *>(_localctx)->exprContext->node))});
                
      break;
    }

    }
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- VarDefsContext ------------------------------------------------------------------

MDParser::VarDefsContext::VarDefsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::VarDefsContext::INT() {
  return getToken(MDParser::INT, 0);
}

MDParser::VarDefContext* MDParser::VarDefsContext::varDef() {
  return getRuleContext<MDParser::VarDefContext>(0);
}

MDParser::VarDefsContext* MDParser::VarDefsContext::varDefs() {
  return getRuleContext<MDParser::VarDefsContext>(0);
}


size_t MDParser::VarDefsContext::getRuleIndex() const {
  return MDParser::RuleVarDefs;
}



MDParser::VarDefsContext* MDParser::varDefs() {
   return varDefs(0);
}

MDParser::VarDefsContext* MDParser::varDefs(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::VarDefsContext *_localctx = _tracker.createInstance<VarDefsContext>(_ctx, parentState);
  MDParser::VarDefsContext *previousContext = _localctx;
  size_t startState = 26;
  enterRecursionRule(_localctx, 26, MDParser::RuleVarDefs, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(330);
    match(MDParser::INT);
    setState(331);
    dynamic_cast<VarDefsContext *>(_localctx)->varDefContext = varDef();

                dynamic_cast<VarDefsContext *>(_localctx)->nodes =  {dynamic_cast<VarDefsContext *>(_localctx)->varDefContext->node};
              
    _ctx->stop = _input->LT(-1);
    setState(341);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<VarDefsContext>(parentContext, parentState);
        _localctx->part = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleVarDefs);
        setState(334);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(335);
        match(MDParser::COMMA);
        setState(336);
        dynamic_cast<VarDefsContext *>(_localctx)->varDefContext = varDef();

                              dynamic_cast<VarDefsContext *>(_localctx)->nodes =  dynamic_cast<VarDefsContext *>(_localctx)->part->nodes;
                              _localctx->nodes.push_back(dynamic_cast<VarDefsContext *>(_localctx)->varDefContext->node);
                             
      }
      setState(343);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 14, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

//----------------- ArgContext ------------------------------------------------------------------

MDParser::ArgContext::ArgContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

tree::TerminalNode* MDParser::ArgContext::INT() {
  return getToken(MDParser::INT, 0);
}

tree::TerminalNode* MDParser::ArgContext::Identifier() {
  return getToken(MDParser::Identifier, 0);
}


size_t MDParser::ArgContext::getRuleIndex() const {
  return MDParser::RuleArg;
}


MDParser::ArgContext* MDParser::arg() {
  ArgContext *_localctx = _tracker.createInstance<ArgContext>(_ctx, getState());
  enterRule(_localctx, 28, MDParser::RuleArg);

  auto onExit = finally([=] {
    exitRule();
  });
  try {
    enterOuterAlt(_localctx, 1);
    setState(344);
    match(MDParser::INT);
    setState(345);
    dynamic_cast<ArgContext *>(_localctx)->identifierToken = match(MDParser::Identifier);

                dynamic_cast<ArgContext *>(_localctx)->node =  std::make_pair(ExprType::Int, (dynamic_cast<ArgContext *>(_localctx)->identifierToken != nullptr ? dynamic_cast<ArgContext *>(_localctx)->identifierToken->getText() : ""));
              
   
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }

  return _localctx;
}

//----------------- ArgsContext ------------------------------------------------------------------

MDParser::ArgsContext::ArgsContext(ParserRuleContext *parent, size_t invokingState)
  : ParserRuleContext(parent, invokingState) {
}

MDParser::ArgContext* MDParser::ArgsContext::arg() {
  return getRuleContext<MDParser::ArgContext>(0);
}

MDParser::ArgsContext* MDParser::ArgsContext::args() {
  return getRuleContext<MDParser::ArgsContext>(0);
}


size_t MDParser::ArgsContext::getRuleIndex() const {
  return MDParser::RuleArgs;
}



MDParser::ArgsContext* MDParser::args() {
   return args(0);
}

MDParser::ArgsContext* MDParser::args(int precedence) {
  ParserRuleContext *parentContext = _ctx;
  size_t parentState = getState();
  MDParser::ArgsContext *_localctx = _tracker.createInstance<ArgsContext>(_ctx, parentState);
  MDParser::ArgsContext *previousContext = _localctx;
  size_t startState = 30;
  enterRecursionRule(_localctx, 30, MDParser::RuleArgs, precedence);

    

  auto onExit = finally([=] {
    unrollRecursionContexts(parentContext);
  });
  try {
    size_t alt;
    enterOuterAlt(_localctx, 1);
    setState(352);
    _errHandler->sync(this);
    switch (getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 15, _ctx)) {
    case 1: {
      break;
    }

    case 2: {
      setState(349);
      dynamic_cast<ArgsContext *>(_localctx)->argContext = arg();

                  dynamic_cast<ArgsContext *>(_localctx)->nodes =  {dynamic_cast<ArgsContext *>(_localctx)->argContext->node};
                
      break;
    }

    }
    _ctx->stop = _input->LT(-1);
    setState(361);
    _errHandler->sync(this);
    alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx);
    while (alt != 2 && alt != atn::ATN::INVALID_ALT_NUMBER) {
      if (alt == 1) {
        if (!_parseListeners.empty())
          triggerExitRuleEvent();
        previousContext = _localctx;
        _localctx = _tracker.createInstance<ArgsContext>(parentContext, parentState);
        _localctx->part = previousContext;
        pushNewRecursionContext(_localctx, startState, RuleArgs);
        setState(354);

        if (!(precpred(_ctx, 1))) throw FailedPredicateException(this, "precpred(_ctx, 1)");
        setState(355);
        match(MDParser::COMMA);
        setState(356);
        dynamic_cast<ArgsContext *>(_localctx)->argContext = arg();

                              dynamic_cast<ArgsContext *>(_localctx)->nodes =  dynamic_cast<ArgsContext *>(_localctx)->part->nodes;
                              _localctx->nodes.push_back(dynamic_cast<ArgsContext *>(_localctx)->argContext->node);
                             
      }
      setState(363);
      _errHandler->sync(this);
      alt = getInterpreter<atn::ParserATNSimulator>()->adaptivePredict(_input, 16, _ctx);
    }
  }
  catch (RecognitionException &e) {
    _errHandler->reportError(this, e);
    _localctx->exception = std::current_exception();
    _errHandler->recover(this, _localctx->exception);
  }
  return _localctx;
}

bool MDParser::sempred(RuleContext *context, size_t ruleIndex, size_t predicateIndex) {
  switch (ruleIndex) {
    case 1: return funcsSempred(dynamic_cast<FuncsContext *>(context), predicateIndex);
    case 5: return stmtSeqSempred(dynamic_cast<StmtSeqContext *>(context), predicateIndex);
    case 7: return exprSempred(dynamic_cast<ExprContext *>(context), predicateIndex);
    case 8: return exprsSempred(dynamic_cast<ExprsContext *>(context), predicateIndex);
    case 11: return varsSempred(dynamic_cast<VarsContext *>(context), predicateIndex);
    case 13: return varDefsSempred(dynamic_cast<VarDefsContext *>(context), predicateIndex);
    case 15: return argsSempred(dynamic_cast<ArgsContext *>(context), predicateIndex);

  default:
    break;
  }
  return true;
}

bool MDParser::funcsSempred(FuncsContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 0: return precpred(_ctx, 3);
    case 1: return precpred(_ctx, 2);
    case 2: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool MDParser::stmtSeqSempred(StmtSeqContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 3: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool MDParser::exprSempred(ExprContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 4: return precpred(_ctx, 12);
    case 5: return precpred(_ctx, 11);
    case 6: return precpred(_ctx, 10);
    case 7: return precpred(_ctx, 9);
    case 8: return precpred(_ctx, 8);
    case 9: return precpred(_ctx, 7);
    case 10: return precpred(_ctx, 6);
    case 11: return precpred(_ctx, 5);
    case 12: return precpred(_ctx, 4);
    case 13: return precpred(_ctx, 3);
    case 14: return precpred(_ctx, 2);

  default:
    break;
  }
  return true;
}

bool MDParser::exprsSempred(ExprsContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 15: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool MDParser::varsSempred(VarsContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 16: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool MDParser::varDefsSempred(VarDefsContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 17: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

bool MDParser::argsSempred(ArgsContext *_localctx, size_t predicateIndex) {
  switch (predicateIndex) {
    case 18: return precpred(_ctx, 1);

  default:
    break;
  }
  return true;
}

// Static vars and initialization.
std::vector<dfa::DFA> MDParser::_decisionToDFA;
atn::PredictionContextCache MDParser::_sharedContextCache;

// We own the ATN which in turn owns the ATN states.
atn::ATN MDParser::_atn;
std::vector<uint16_t> MDParser::_serializedATN;

std::vector<std::string> MDParser::_ruleNames = {
  "program", "funcs", "func", "funcDec", "gVarDef", "stmtSeq", "stmt", "expr", 
  "exprs", "mayExpr", "var", "vars", "varDef", "varDefs", "arg", "args"
};

std::vector<std::string> MDParser::_literalNames = {
  "", "", "'if'", "'else'", "'do'", "'while'", "'for'", "'return'", "'break'", 
  "'continue'", "'int'", "", "", "'='", "'+'", "'-'", "'*'", "'/'", "'%'", 
  "'!'", "'~'", "'&'", "'^'", "'|'", "'<<'", "'>>'", "'=='", "'!='", "'<'", 
  "'>'", "'<='", "'>='", "'&&'", "'||'", "':'", "'?'", "';'", "'('", "')'", 
  "'{'", "'}'", "','"
};

std::vector<std::string> MDParser::_symbolicNames = {
  "", "WhiteSpaces", "IF", "ELSE", "DO", "WHILE", "FOR", "RETURN", "BREAK", 
  "CONTINUE", "INT", "Integer", "Identifier", "ASSIGN", "PLUS", "MINUS", 
  "STAR", "SLASH", "PERCENT", "NOT", "TILDE", "AND", "HAT", "OR", "SL", 
  "SR", "EQ", "NE", "LT", "GT", "LE", "GE", "LAND", "LOR", "COLON", "QUESTION", 
  "SEMICOLON", "LPAREN", "RPAREN", "LBRACK", "RBRACK", "COMMA"
};

dfa::Vocabulary MDParser::_vocabulary(_literalNames, _symbolicNames);

std::vector<std::string> MDParser::_tokenNames;

MDParser::Initializer::Initializer() {
	for (size_t i = 0; i < _symbolicNames.size(); ++i) {
		std::string name = _vocabulary.getLiteralName(i);
		if (name.empty()) {
			name = _vocabulary.getSymbolicName(i);
		}

		if (name.empty()) {
			_tokenNames.push_back("<INVALID>");
		} else {
      _tokenNames.push_back(name);
    }
	}

  _serializedATN = {
    0x3, 0x608b, 0xa72a, 0x8133, 0xb9ed, 0x417c, 0x3be7, 0x7786, 0x5964, 
    0x3, 0x2b, 0x16f, 0x4, 0x2, 0x9, 0x2, 0x4, 0x3, 0x9, 0x3, 0x4, 0x4, 
    0x9, 0x4, 0x4, 0x5, 0x9, 0x5, 0x4, 0x6, 0x9, 0x6, 0x4, 0x7, 0x9, 0x7, 
    0x4, 0x8, 0x9, 0x8, 0x4, 0x9, 0x9, 0x9, 0x4, 0xa, 0x9, 0xa, 0x4, 0xb, 
    0x9, 0xb, 0x4, 0xc, 0x9, 0xc, 0x4, 0xd, 0x9, 0xd, 0x4, 0xe, 0x9, 0xe, 
    0x4, 0xf, 0x9, 0xf, 0x4, 0x10, 0x9, 0x10, 0x4, 0x11, 0x9, 0x11, 0x3, 
    0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x2, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 
    0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x3, 0x7, 0x3, 0x36, 0xa, 0x3, 
    0xc, 0x3, 0xe, 0x3, 0x39, 0xb, 0x3, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 0x4, 0x3, 
    0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 0x5, 0x3, 
    0x5, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 
    0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x3, 0x6, 0x5, 0x6, 0x58, 0xa, 0x6, 
    0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 0x3, 0x7, 
    0x7, 0x7, 0x61, 0xa, 0x7, 0xc, 0x7, 0xe, 0x7, 0x64, 0xb, 0x7, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 0x3, 0x8, 
    0x3, 0x8, 0x3, 0x8, 0x5, 0x8, 0xb6, 0xa, 0x8, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 
    0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x5, 0x9, 0xda, 0xa, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 0x3, 0x9, 
    0x7, 0x9, 0x115, 0xa, 0x9, 0xc, 0x9, 0xe, 0x9, 0x118, 0xb, 0x9, 0x3, 
    0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x5, 0xa, 0x11e, 0xa, 0xa, 0x3, 0xa, 
    0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x3, 0xa, 0x7, 0xa, 0x125, 0xa, 0xa, 0xc, 
    0xa, 0xe, 0xa, 0x128, 0xb, 0xa, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 0x3, 0xb, 
    0x5, 0xb, 0x12e, 0xa, 0xb, 0x3, 0xc, 0x3, 0xc, 0x3, 0xc, 0x3, 0xd, 0x3, 
    0xd, 0x3, 0xd, 0x3, 0xd, 0x5, 0xd, 0x137, 0xa, 0xd, 0x3, 0xd, 0x3, 0xd, 
    0x3, 0xd, 0x3, 0xd, 0x3, 0xd, 0x7, 0xd, 0x13e, 0xa, 0xd, 0xc, 0xd, 0xe, 
    0xd, 0x141, 0xb, 0xd, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 0x3, 0xe, 
    0x3, 0xe, 0x3, 0xe, 0x5, 0xe, 0x14a, 0xa, 0xe, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 0xf, 0x3, 
    0xf, 0x7, 0xf, 0x156, 0xa, 0xf, 0xc, 0xf, 0xe, 0xf, 0x159, 0xb, 0xf, 
    0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x10, 0x3, 0x11, 0x3, 0x11, 0x3, 
    0x11, 0x3, 0x11, 0x5, 0x11, 0x163, 0xa, 0x11, 0x3, 0x11, 0x3, 0x11, 
    0x3, 0x11, 0x3, 0x11, 0x3, 0x11, 0x7, 0x11, 0x16a, 0xa, 0x11, 0xc, 0x11, 
    0xe, 0x11, 0x16d, 0xb, 0x11, 0x3, 0x11, 0x2, 0x9, 0x4, 0xc, 0x10, 0x12, 
    0x18, 0x1c, 0x20, 0x12, 0x2, 0x4, 0x6, 0x8, 0xa, 0xc, 0xe, 0x10, 0x12, 
    0x14, 0x16, 0x18, 0x1a, 0x1c, 0x1e, 0x20, 0x2, 0x7, 0x3, 0x2, 0x12, 
    0x14, 0x3, 0x2, 0x10, 0x11, 0x3, 0x2, 0x1a, 0x1b, 0x3, 0x2, 0x1e, 0x21, 
    0x3, 0x2, 0x1c, 0x1d, 0x2, 0x18a, 0x2, 0x22, 0x3, 0x2, 0x2, 0x2, 0x4, 
    0x26, 0x3, 0x2, 0x2, 0x2, 0x6, 0x3a, 0x3, 0x2, 0x2, 0x2, 0x8, 0x44, 
    0x3, 0x2, 0x2, 0x2, 0xa, 0x57, 0x3, 0x2, 0x2, 0x2, 0xc, 0x59, 0x3, 0x2, 
    0x2, 0x2, 0xe, 0xb5, 0x3, 0x2, 0x2, 0x2, 0x10, 0xd9, 0x3, 0x2, 0x2, 
    0x2, 0x12, 0x11d, 0x3, 0x2, 0x2, 0x2, 0x14, 0x12d, 0x3, 0x2, 0x2, 0x2, 
    0x16, 0x12f, 0x3, 0x2, 0x2, 0x2, 0x18, 0x136, 0x3, 0x2, 0x2, 0x2, 0x1a, 
    0x149, 0x3, 0x2, 0x2, 0x2, 0x1c, 0x14b, 0x3, 0x2, 0x2, 0x2, 0x1e, 0x15a, 
    0x3, 0x2, 0x2, 0x2, 0x20, 0x162, 0x3, 0x2, 0x2, 0x2, 0x22, 0x23, 0x5, 
    0x4, 0x3, 0x2, 0x23, 0x24, 0x7, 0x2, 0x2, 0x3, 0x24, 0x25, 0x8, 0x2, 
    0x1, 0x2, 0x25, 0x3, 0x3, 0x2, 0x2, 0x2, 0x26, 0x27, 0x8, 0x3, 0x1, 
    0x2, 0x27, 0x28, 0x8, 0x3, 0x1, 0x2, 0x28, 0x37, 0x3, 0x2, 0x2, 0x2, 
    0x29, 0x2a, 0xc, 0x5, 0x2, 0x2, 0x2a, 0x2b, 0x5, 0x6, 0x4, 0x2, 0x2b, 
    0x2c, 0x8, 0x3, 0x1, 0x2, 0x2c, 0x36, 0x3, 0x2, 0x2, 0x2, 0x2d, 0x2e, 
    0xc, 0x4, 0x2, 0x2, 0x2e, 0x2f, 0x5, 0x8, 0x5, 0x2, 0x2f, 0x30, 0x8, 
    0x3, 0x1, 0x2, 0x30, 0x36, 0x3, 0x2, 0x2, 0x2, 0x31, 0x32, 0xc, 0x3, 
    0x2, 0x2, 0x32, 0x33, 0x5, 0xa, 0x6, 0x2, 0x33, 0x34, 0x8, 0x3, 0x1, 
    0x2, 0x34, 0x36, 0x3, 0x2, 0x2, 0x2, 0x35, 0x29, 0x3, 0x2, 0x2, 0x2, 
    0x35, 0x2d, 0x3, 0x2, 0x2, 0x2, 0x35, 0x31, 0x3, 0x2, 0x2, 0x2, 0x36, 
    0x39, 0x3, 0x2, 0x2, 0x2, 0x37, 0x35, 0x3, 0x2, 0x2, 0x2, 0x37, 0x38, 
    0x3, 0x2, 0x2, 0x2, 0x38, 0x5, 0x3, 0x2, 0x2, 0x2, 0x39, 0x37, 0x3, 
    0x2, 0x2, 0x2, 0x3a, 0x3b, 0x7, 0xc, 0x2, 0x2, 0x3b, 0x3c, 0x7, 0xe, 
    0x2, 0x2, 0x3c, 0x3d, 0x7, 0x27, 0x2, 0x2, 0x3d, 0x3e, 0x5, 0x20, 0x11, 
    0x2, 0x3e, 0x3f, 0x7, 0x28, 0x2, 0x2, 0x3f, 0x40, 0x7, 0x29, 0x2, 0x2, 
    0x40, 0x41, 0x5, 0xc, 0x7, 0x2, 0x41, 0x42, 0x7, 0x2a, 0x2, 0x2, 0x42, 
    0x43, 0x8, 0x4, 0x1, 0x2, 0x43, 0x7, 0x3, 0x2, 0x2, 0x2, 0x44, 0x45, 
    0x7, 0xc, 0x2, 0x2, 0x45, 0x46, 0x7, 0xe, 0x2, 0x2, 0x46, 0x47, 0x7, 
    0x27, 0x2, 0x2, 0x47, 0x48, 0x5, 0x20, 0x11, 0x2, 0x48, 0x49, 0x7, 0x28, 
    0x2, 0x2, 0x49, 0x4a, 0x7, 0x26, 0x2, 0x2, 0x4a, 0x4b, 0x8, 0x5, 0x1, 
    0x2, 0x4b, 0x9, 0x3, 0x2, 0x2, 0x2, 0x4c, 0x4d, 0x7, 0xc, 0x2, 0x2, 
    0x4d, 0x4e, 0x7, 0xe, 0x2, 0x2, 0x4e, 0x4f, 0x7, 0x26, 0x2, 0x2, 0x4f, 
    0x58, 0x8, 0x6, 0x1, 0x2, 0x50, 0x51, 0x7, 0xc, 0x2, 0x2, 0x51, 0x52, 
    0x7, 0xe, 0x2, 0x2, 0x52, 0x53, 0x7, 0xf, 0x2, 0x2, 0x53, 0x54, 0x5, 
    0x10, 0x9, 0x2, 0x54, 0x55, 0x7, 0x26, 0x2, 0x2, 0x55, 0x56, 0x8, 0x6, 
    0x1, 0x2, 0x56, 0x58, 0x3, 0x2, 0x2, 0x2, 0x57, 0x4c, 0x3, 0x2, 0x2, 
    0x2, 0x57, 0x50, 0x3, 0x2, 0x2, 0x2, 0x58, 0xb, 0x3, 0x2, 0x2, 0x2, 
    0x59, 0x5a, 0x8, 0x7, 0x1, 0x2, 0x5a, 0x5b, 0x8, 0x7, 0x1, 0x2, 0x5b, 
    0x62, 0x3, 0x2, 0x2, 0x2, 0x5c, 0x5d, 0xc, 0x3, 0x2, 0x2, 0x5d, 0x5e, 
    0x5, 0xe, 0x8, 0x2, 0x5e, 0x5f, 0x8, 0x7, 0x1, 0x2, 0x5f, 0x61, 0x3, 
    0x2, 0x2, 0x2, 0x60, 0x5c, 0x3, 0x2, 0x2, 0x2, 0x61, 0x64, 0x3, 0x2, 
    0x2, 0x2, 0x62, 0x60, 0x3, 0x2, 0x2, 0x2, 0x62, 0x63, 0x3, 0x2, 0x2, 
    0x2, 0x63, 0xd, 0x3, 0x2, 0x2, 0x2, 0x64, 0x62, 0x3, 0x2, 0x2, 0x2, 
    0x65, 0x66, 0x5, 0x10, 0x9, 0x2, 0x66, 0x67, 0x7, 0x26, 0x2, 0x2, 0x67, 
    0x68, 0x8, 0x8, 0x1, 0x2, 0x68, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x69, 0x6a, 
    0x5, 0x1c, 0xf, 0x2, 0x6a, 0x6b, 0x7, 0x26, 0x2, 0x2, 0x6b, 0x6c, 0x8, 
    0x8, 0x1, 0x2, 0x6c, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x6d, 0x6e, 0x7, 0x4, 
    0x2, 0x2, 0x6e, 0x6f, 0x7, 0x27, 0x2, 0x2, 0x6f, 0x70, 0x5, 0x10, 0x9, 
    0x2, 0x70, 0x71, 0x7, 0x28, 0x2, 0x2, 0x71, 0x72, 0x5, 0xe, 0x8, 0x2, 
    0x72, 0x73, 0x8, 0x8, 0x1, 0x2, 0x73, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x74, 
    0x75, 0x7, 0x4, 0x2, 0x2, 0x75, 0x76, 0x7, 0x27, 0x2, 0x2, 0x76, 0x77, 
    0x5, 0x10, 0x9, 0x2, 0x77, 0x78, 0x7, 0x28, 0x2, 0x2, 0x78, 0x79, 0x5, 
    0xe, 0x8, 0x2, 0x79, 0x7a, 0x7, 0x5, 0x2, 0x2, 0x7a, 0x7b, 0x5, 0xe, 
    0x8, 0x2, 0x7b, 0x7c, 0x8, 0x8, 0x1, 0x2, 0x7c, 0xb6, 0x3, 0x2, 0x2, 
    0x2, 0x7d, 0x7e, 0x7, 0x7, 0x2, 0x2, 0x7e, 0x7f, 0x7, 0x27, 0x2, 0x2, 
    0x7f, 0x80, 0x5, 0x10, 0x9, 0x2, 0x80, 0x81, 0x7, 0x28, 0x2, 0x2, 0x81, 
    0x82, 0x5, 0xe, 0x8, 0x2, 0x82, 0x83, 0x8, 0x8, 0x1, 0x2, 0x83, 0xb6, 
    0x3, 0x2, 0x2, 0x2, 0x84, 0x85, 0x7, 0x6, 0x2, 0x2, 0x85, 0x86, 0x5, 
    0xe, 0x8, 0x2, 0x86, 0x87, 0x7, 0x7, 0x2, 0x2, 0x87, 0x88, 0x7, 0x27, 
    0x2, 0x2, 0x88, 0x89, 0x5, 0x10, 0x9, 0x2, 0x89, 0x8a, 0x7, 0x28, 0x2, 
    0x2, 0x8a, 0x8b, 0x7, 0x26, 0x2, 0x2, 0x8b, 0x8c, 0x8, 0x8, 0x1, 0x2, 
    0x8c, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x8d, 0x8e, 0x7, 0x8, 0x2, 0x2, 0x8e, 
    0x8f, 0x7, 0x27, 0x2, 0x2, 0x8f, 0x90, 0x5, 0x14, 0xb, 0x2, 0x90, 0x91, 
    0x7, 0x26, 0x2, 0x2, 0x91, 0x92, 0x5, 0x14, 0xb, 0x2, 0x92, 0x93, 0x7, 
    0x26, 0x2, 0x2, 0x93, 0x94, 0x5, 0x14, 0xb, 0x2, 0x94, 0x95, 0x7, 0x28, 
    0x2, 0x2, 0x95, 0x96, 0x5, 0xe, 0x8, 0x2, 0x96, 0x97, 0x8, 0x8, 0x1, 
    0x2, 0x97, 0xb6, 0x3, 0x2, 0x2, 0x2, 0x98, 0x99, 0x7, 0x8, 0x2, 0x2, 
    0x99, 0x9a, 0x7, 0x27, 0x2, 0x2, 0x9a, 0x9b, 0x5, 0x1c, 0xf, 0x2, 0x9b, 
    0x9c, 0x7, 0x26, 0x2, 0x2, 0x9c, 0x9d, 0x5, 0x14, 0xb, 0x2, 0x9d, 0x9e, 
    0x7, 0x26, 0x2, 0x2, 0x9e, 0x9f, 0x5, 0x14, 0xb, 0x2, 0x9f, 0xa0, 0x7, 
    0x28, 0x2, 0x2, 0xa0, 0xa1, 0x5, 0xe, 0x8, 0x2, 0xa1, 0xa2, 0x8, 0x8, 
    0x1, 0x2, 0xa2, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xa3, 0xa4, 0x7, 0x9, 0x2, 
    0x2, 0xa4, 0xa5, 0x5, 0x10, 0x9, 0x2, 0xa5, 0xa6, 0x7, 0x26, 0x2, 0x2, 
    0xa6, 0xa7, 0x8, 0x8, 0x1, 0x2, 0xa7, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xa8, 
    0xa9, 0x7, 0xa, 0x2, 0x2, 0xa9, 0xaa, 0x7, 0x26, 0x2, 0x2, 0xaa, 0xb6, 
    0x8, 0x8, 0x1, 0x2, 0xab, 0xac, 0x7, 0xb, 0x2, 0x2, 0xac, 0xad, 0x7, 
    0x26, 0x2, 0x2, 0xad, 0xb6, 0x8, 0x8, 0x1, 0x2, 0xae, 0xaf, 0x7, 0x26, 
    0x2, 0x2, 0xaf, 0xb6, 0x8, 0x8, 0x1, 0x2, 0xb0, 0xb1, 0x7, 0x29, 0x2, 
    0x2, 0xb1, 0xb2, 0x5, 0xc, 0x7, 0x2, 0xb2, 0xb3, 0x7, 0x2a, 0x2, 0x2, 
    0xb3, 0xb4, 0x8, 0x8, 0x1, 0x2, 0xb4, 0xb6, 0x3, 0x2, 0x2, 0x2, 0xb5, 
    0x65, 0x3, 0x2, 0x2, 0x2, 0xb5, 0x69, 0x3, 0x2, 0x2, 0x2, 0xb5, 0x6d, 
    0x3, 0x2, 0x2, 0x2, 0xb5, 0x74, 0x3, 0x2, 0x2, 0x2, 0xb5, 0x7d, 0x3, 
    0x2, 0x2, 0x2, 0xb5, 0x84, 0x3, 0x2, 0x2, 0x2, 0xb5, 0x8d, 0x3, 0x2, 
    0x2, 0x2, 0xb5, 0x98, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xa3, 0x3, 0x2, 0x2, 
    0x2, 0xb5, 0xa8, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xab, 0x3, 0x2, 0x2, 0x2, 
    0xb5, 0xae, 0x3, 0x2, 0x2, 0x2, 0xb5, 0xb0, 0x3, 0x2, 0x2, 0x2, 0xb6, 
    0xf, 0x3, 0x2, 0x2, 0x2, 0xb7, 0xb8, 0x8, 0x9, 0x1, 0x2, 0xb8, 0xb9, 
    0x7, 0xd, 0x2, 0x2, 0xb9, 0xda, 0x8, 0x9, 0x1, 0x2, 0xba, 0xbb, 0x5, 
    0x16, 0xc, 0x2, 0xbb, 0xbc, 0x8, 0x9, 0x1, 0x2, 0xbc, 0xda, 0x3, 0x2, 
    0x2, 0x2, 0xbd, 0xbe, 0x7, 0xe, 0x2, 0x2, 0xbe, 0xbf, 0x7, 0x27, 0x2, 
    0x2, 0xbf, 0xc0, 0x5, 0x12, 0xa, 0x2, 0xc0, 0xc1, 0x7, 0x28, 0x2, 0x2, 
    0xc1, 0xc2, 0x8, 0x9, 0x1, 0x2, 0xc2, 0xda, 0x3, 0x2, 0x2, 0x2, 0xc3, 
    0xc4, 0x7, 0x27, 0x2, 0x2, 0xc4, 0xc5, 0x5, 0x10, 0x9, 0x2, 0xc5, 0xc6, 
    0x7, 0x28, 0x2, 0x2, 0xc6, 0xc7, 0x8, 0x9, 0x1, 0x2, 0xc7, 0xda, 0x3, 
    0x2, 0x2, 0x2, 0xc8, 0xc9, 0x7, 0x11, 0x2, 0x2, 0xc9, 0xca, 0x5, 0x10, 
    0x9, 0x11, 0xca, 0xcb, 0x8, 0x9, 0x1, 0x2, 0xcb, 0xda, 0x3, 0x2, 0x2, 
    0x2, 0xcc, 0xcd, 0x7, 0x15, 0x2, 0x2, 0xcd, 0xce, 0x5, 0x10, 0x9, 0x10, 
    0xce, 0xcf, 0x8, 0x9, 0x1, 0x2, 0xcf, 0xda, 0x3, 0x2, 0x2, 0x2, 0xd0, 
    0xd1, 0x7, 0x16, 0x2, 0x2, 0xd1, 0xd2, 0x5, 0x10, 0x9, 0xf, 0xd2, 0xd3, 
    0x8, 0x9, 0x1, 0x2, 0xd3, 0xda, 0x3, 0x2, 0x2, 0x2, 0xd4, 0xd5, 0x7, 
    0xe, 0x2, 0x2, 0xd5, 0xd6, 0x7, 0xf, 0x2, 0x2, 0xd6, 0xd7, 0x5, 0x10, 
    0x9, 0x3, 0xd7, 0xd8, 0x8, 0x9, 0x1, 0x2, 0xd8, 0xda, 0x3, 0x2, 0x2, 
    0x2, 0xd9, 0xb7, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xba, 0x3, 0x2, 0x2, 0x2, 
    0xd9, 0xbd, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xc3, 0x3, 0x2, 0x2, 0x2, 0xd9, 
    0xc8, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xcc, 0x3, 0x2, 0x2, 0x2, 0xd9, 0xd0, 
    0x3, 0x2, 0x2, 0x2, 0xd9, 0xd4, 0x3, 0x2, 0x2, 0x2, 0xda, 0x116, 0x3, 
    0x2, 0x2, 0x2, 0xdb, 0xdc, 0xc, 0xe, 0x2, 0x2, 0xdc, 0xdd, 0x9, 0x2, 
    0x2, 0x2, 0xdd, 0xde, 0x5, 0x10, 0x9, 0xf, 0xde, 0xdf, 0x8, 0x9, 0x1, 
    0x2, 0xdf, 0x115, 0x3, 0x2, 0x2, 0x2, 0xe0, 0xe1, 0xc, 0xd, 0x2, 0x2, 
    0xe1, 0xe2, 0x9, 0x3, 0x2, 0x2, 0xe2, 0xe3, 0x5, 0x10, 0x9, 0xe, 0xe3, 
    0xe4, 0x8, 0x9, 0x1, 0x2, 0xe4, 0x115, 0x3, 0x2, 0x2, 0x2, 0xe5, 0xe6, 
    0xc, 0xc, 0x2, 0x2, 0xe6, 0xe7, 0x9, 0x4, 0x2, 0x2, 0xe7, 0xe8, 0x5, 
    0x10, 0x9, 0xd, 0xe8, 0xe9, 0x8, 0x9, 0x1, 0x2, 0xe9, 0x115, 0x3, 0x2, 
    0x2, 0x2, 0xea, 0xeb, 0xc, 0xb, 0x2, 0x2, 0xeb, 0xec, 0x9, 0x5, 0x2, 
    0x2, 0xec, 0xed, 0x5, 0x10, 0x9, 0xc, 0xed, 0xee, 0x8, 0x9, 0x1, 0x2, 
    0xee, 0x115, 0x3, 0x2, 0x2, 0x2, 0xef, 0xf0, 0xc, 0xa, 0x2, 0x2, 0xf0, 
    0xf1, 0x9, 0x6, 0x2, 0x2, 0xf1, 0xf2, 0x5, 0x10, 0x9, 0xb, 0xf2, 0xf3, 
    0x8, 0x9, 0x1, 0x2, 0xf3, 0x115, 0x3, 0x2, 0x2, 0x2, 0xf4, 0xf5, 0xc, 
    0x9, 0x2, 0x2, 0xf5, 0xf6, 0x7, 0x17, 0x2, 0x2, 0xf6, 0xf7, 0x5, 0x10, 
    0x9, 0xa, 0xf7, 0xf8, 0x8, 0x9, 0x1, 0x2, 0xf8, 0x115, 0x3, 0x2, 0x2, 
    0x2, 0xf9, 0xfa, 0xc, 0x8, 0x2, 0x2, 0xfa, 0xfb, 0x7, 0x18, 0x2, 0x2, 
    0xfb, 0xfc, 0x5, 0x10, 0x9, 0x9, 0xfc, 0xfd, 0x8, 0x9, 0x1, 0x2, 0xfd, 
    0x115, 0x3, 0x2, 0x2, 0x2, 0xfe, 0xff, 0xc, 0x7, 0x2, 0x2, 0xff, 0x100, 
    0x7, 0x19, 0x2, 0x2, 0x100, 0x101, 0x5, 0x10, 0x9, 0x8, 0x101, 0x102, 
    0x8, 0x9, 0x1, 0x2, 0x102, 0x115, 0x3, 0x2, 0x2, 0x2, 0x103, 0x104, 
    0xc, 0x6, 0x2, 0x2, 0x104, 0x105, 0x7, 0x22, 0x2, 0x2, 0x105, 0x106, 
    0x5, 0x10, 0x9, 0x7, 0x106, 0x107, 0x8, 0x9, 0x1, 0x2, 0x107, 0x115, 
    0x3, 0x2, 0x2, 0x2, 0x108, 0x109, 0xc, 0x5, 0x2, 0x2, 0x109, 0x10a, 
    0x7, 0x23, 0x2, 0x2, 0x10a, 0x10b, 0x5, 0x10, 0x9, 0x6, 0x10b, 0x10c, 
    0x8, 0x9, 0x1, 0x2, 0x10c, 0x115, 0x3, 0x2, 0x2, 0x2, 0x10d, 0x10e, 
    0xc, 0x4, 0x2, 0x2, 0x10e, 0x10f, 0x7, 0x25, 0x2, 0x2, 0x10f, 0x110, 
    0x5, 0x10, 0x9, 0x2, 0x110, 0x111, 0x7, 0x24, 0x2, 0x2, 0x111, 0x112, 
    0x5, 0x10, 0x9, 0x5, 0x112, 0x113, 0x8, 0x9, 0x1, 0x2, 0x113, 0x115, 
    0x3, 0x2, 0x2, 0x2, 0x114, 0xdb, 0x3, 0x2, 0x2, 0x2, 0x114, 0xe0, 0x3, 
    0x2, 0x2, 0x2, 0x114, 0xe5, 0x3, 0x2, 0x2, 0x2, 0x114, 0xea, 0x3, 0x2, 
    0x2, 0x2, 0x114, 0xef, 0x3, 0x2, 0x2, 0x2, 0x114, 0xf4, 0x3, 0x2, 0x2, 
    0x2, 0x114, 0xf9, 0x3, 0x2, 0x2, 0x2, 0x114, 0xfe, 0x3, 0x2, 0x2, 0x2, 
    0x114, 0x103, 0x3, 0x2, 0x2, 0x2, 0x114, 0x108, 0x3, 0x2, 0x2, 0x2, 
    0x114, 0x10d, 0x3, 0x2, 0x2, 0x2, 0x115, 0x118, 0x3, 0x2, 0x2, 0x2, 
    0x116, 0x114, 0x3, 0x2, 0x2, 0x2, 0x116, 0x117, 0x3, 0x2, 0x2, 0x2, 
    0x117, 0x11, 0x3, 0x2, 0x2, 0x2, 0x118, 0x116, 0x3, 0x2, 0x2, 0x2, 0x119, 
    0x11e, 0x8, 0xa, 0x1, 0x2, 0x11a, 0x11b, 0x5, 0x10, 0x9, 0x2, 0x11b, 
    0x11c, 0x8, 0xa, 0x1, 0x2, 0x11c, 0x11e, 0x3, 0x2, 0x2, 0x2, 0x11d, 
    0x119, 0x3, 0x2, 0x2, 0x2, 0x11d, 0x11a, 0x3, 0x2, 0x2, 0x2, 0x11e, 
    0x126, 0x3, 0x2, 0x2, 0x2, 0x11f, 0x120, 0xc, 0x3, 0x2, 0x2, 0x120, 
    0x121, 0x7, 0x2b, 0x2, 0x2, 0x121, 0x122, 0x5, 0x10, 0x9, 0x2, 0x122, 
    0x123, 0x8, 0xa, 0x1, 0x2, 0x123, 0x125, 0x3, 0x2, 0x2, 0x2, 0x124, 
    0x11f, 0x3, 0x2, 0x2, 0x2, 0x125, 0x128, 0x3, 0x2, 0x2, 0x2, 0x126, 
    0x124, 0x3, 0x2, 0x2, 0x2, 0x126, 0x127, 0x3, 0x2, 0x2, 0x2, 0x127, 
    0x13, 0x3, 0x2, 0x2, 0x2, 0x128, 0x126, 0x3, 0x2, 0x2, 0x2, 0x129, 0x12e, 
    0x3, 0x2, 0x2, 0x2, 0x12a, 0x12b, 0x5, 0x10, 0x9, 0x2, 0x12b, 0x12c, 
    0x8, 0xb, 0x1, 0x2, 0x12c, 0x12e, 0x3, 0x2, 0x2, 0x2, 0x12d, 0x129, 
    0x3, 0x2, 0x2, 0x2, 0x12d, 0x12a, 0x3, 0x2, 0x2, 0x2, 0x12e, 0x15, 0x3, 
    0x2, 0x2, 0x2, 0x12f, 0x130, 0x7, 0xe, 0x2, 0x2, 0x130, 0x131, 0x8, 
    0xc, 0x1, 0x2, 0x131, 0x17, 0x3, 0x2, 0x2, 0x2, 0x132, 0x137, 0x8, 0xd, 
    0x1, 0x2, 0x133, 0x134, 0x5, 0x16, 0xc, 0x2, 0x134, 0x135, 0x8, 0xd, 
    0x1, 0x2, 0x135, 0x137, 0x3, 0x2, 0x2, 0x2, 0x136, 0x132, 0x3, 0x2, 
    0x2, 0x2, 0x136, 0x133, 0x3, 0x2, 0x2, 0x2, 0x137, 0x13f, 0x3, 0x2, 
    0x2, 0x2, 0x138, 0x139, 0xc, 0x3, 0x2, 0x2, 0x139, 0x13a, 0x7, 0x2b, 
    0x2, 0x2, 0x13a, 0x13b, 0x5, 0x16, 0xc, 0x2, 0x13b, 0x13c, 0x8, 0xd, 
    0x1, 0x2, 0x13c, 0x13e, 0x3, 0x2, 0x2, 0x2, 0x13d, 0x138, 0x3, 0x2, 
    0x2, 0x2, 0x13e, 0x141, 0x3, 0x2, 0x2, 0x2, 0x13f, 0x13d, 0x3, 0x2, 
    0x2, 0x2, 0x13f, 0x140, 0x3, 0x2, 0x2, 0x2, 0x140, 0x19, 0x3, 0x2, 0x2, 
    0x2, 0x141, 0x13f, 0x3, 0x2, 0x2, 0x2, 0x142, 0x143, 0x7, 0xe, 0x2, 
    0x2, 0x143, 0x14a, 0x8, 0xe, 0x1, 0x2, 0x144, 0x145, 0x7, 0xe, 0x2, 
    0x2, 0x145, 0x146, 0x7, 0xf, 0x2, 0x2, 0x146, 0x147, 0x5, 0x10, 0x9, 
    0x2, 0x147, 0x148, 0x8, 0xe, 0x1, 0x2, 0x148, 0x14a, 0x3, 0x2, 0x2, 
    0x2, 0x149, 0x142, 0x3, 0x2, 0x2, 0x2, 0x149, 0x144, 0x3, 0x2, 0x2, 
    0x2, 0x14a, 0x1b, 0x3, 0x2, 0x2, 0x2, 0x14b, 0x14c, 0x8, 0xf, 0x1, 0x2, 
    0x14c, 0x14d, 0x7, 0xc, 0x2, 0x2, 0x14d, 0x14e, 0x5, 0x1a, 0xe, 0x2, 
    0x14e, 0x14f, 0x8, 0xf, 0x1, 0x2, 0x14f, 0x157, 0x3, 0x2, 0x2, 0x2, 
    0x150, 0x151, 0xc, 0x3, 0x2, 0x2, 0x151, 0x152, 0x7, 0x2b, 0x2, 0x2, 
    0x152, 0x153, 0x5, 0x1a, 0xe, 0x2, 0x153, 0x154, 0x8, 0xf, 0x1, 0x2, 
    0x154, 0x156, 0x3, 0x2, 0x2, 0x2, 0x155, 0x150, 0x3, 0x2, 0x2, 0x2, 
    0x156, 0x159, 0x3, 0x2, 0x2, 0x2, 0x157, 0x155, 0x3, 0x2, 0x2, 0x2, 
    0x157, 0x158, 0x3, 0x2, 0x2, 0x2, 0x158, 0x1d, 0x3, 0x2, 0x2, 0x2, 0x159, 
    0x157, 0x3, 0x2, 0x2, 0x2, 0x15a, 0x15b, 0x7, 0xc, 0x2, 0x2, 0x15b, 
    0x15c, 0x7, 0xe, 0x2, 0x2, 0x15c, 0x15d, 0x8, 0x10, 0x1, 0x2, 0x15d, 
    0x1f, 0x3, 0x2, 0x2, 0x2, 0x15e, 0x163, 0x8, 0x11, 0x1, 0x2, 0x15f, 
    0x160, 0x5, 0x1e, 0x10, 0x2, 0x160, 0x161, 0x8, 0x11, 0x1, 0x2, 0x161, 
    0x163, 0x3, 0x2, 0x2, 0x2, 0x162, 0x15e, 0x3, 0x2, 0x2, 0x2, 0x162, 
    0x15f, 0x3, 0x2, 0x2, 0x2, 0x163, 0x16b, 0x3, 0x2, 0x2, 0x2, 0x164, 
    0x165, 0xc, 0x3, 0x2, 0x2, 0x165, 0x166, 0x7, 0x2b, 0x2, 0x2, 0x166, 
    0x167, 0x5, 0x1e, 0x10, 0x2, 0x167, 0x168, 0x8, 0x11, 0x1, 0x2, 0x168, 
    0x16a, 0x3, 0x2, 0x2, 0x2, 0x169, 0x164, 0x3, 0x2, 0x2, 0x2, 0x16a, 
    0x16d, 0x3, 0x2, 0x2, 0x2, 0x16b, 0x169, 0x3, 0x2, 0x2, 0x2, 0x16b, 
    0x16c, 0x3, 0x2, 0x2, 0x2, 0x16c, 0x21, 0x3, 0x2, 0x2, 0x2, 0x16d, 0x16b, 
    0x3, 0x2, 0x2, 0x2, 0x13, 0x35, 0x37, 0x57, 0x62, 0xb5, 0xd9, 0x114, 
    0x116, 0x11d, 0x126, 0x12d, 0x136, 0x13f, 0x149, 0x157, 0x162, 0x16b, 
  };

  atn::ATNDeserializer deserializer;
  _atn = deserializer.deserialize(_serializedATN);

  size_t count = _atn.getNumberOfDecisions();
  _decisionToDFA.reserve(count);
  for (size_t i = 0; i < count; i++) { 
    _decisionToDFA.emplace_back(_atn.getDecisionState(i), i);
  }
}

MDParser::Initializer MDParser::_init;
