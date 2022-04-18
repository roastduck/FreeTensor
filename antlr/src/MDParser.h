
// Generated from MDParser.g4 by ANTLR 4.7.1

#pragma once


#include "antlr4-runtime.h"



#include <string>
#include <vector>

#include "ASTNode.h"





class  MDParser : public antlr4::Parser {
public:
  enum {
    WhiteSpaces = 1, IF = 2, ELSE = 3, DO = 4, WHILE = 5, FOR = 6, RETURN = 7, 
    BREAK = 8, CONTINUE = 9, INT = 10, Integer = 11, Identifier = 12, ASSIGN = 13, 
    PLUS = 14, MINUS = 15, STAR = 16, SLASH = 17, PERCENT = 18, NOT = 19, 
    TILDE = 20, AND = 21, HAT = 22, OR = 23, SL = 24, SR = 25, EQ = 26, 
    NE = 27, LT = 28, GT = 29, LE = 30, GE = 31, LAND = 32, LOR = 33, COLON = 34, 
    QUESTION = 35, SEMICOLON = 36, LPAREN = 37, RPAREN = 38, LBRACK = 39, 
    RBRACK = 40, COMMA = 41
  };

  enum {
    RuleProgram = 0, RuleFuncs = 1, RuleFunc = 2, RuleFuncDec = 3, RuleGVarDef = 4, 
    RuleStmtSeq = 5, RuleStmt = 6, RuleExpr = 7, RuleExprs = 8, RuleMayExpr = 9, 
    RuleVar = 10, RuleVars = 11, RuleVarDef = 12, RuleVarDefs = 13, RuleArg = 14, 
    RuleArgs = 15
  };

  MDParser(antlr4::TokenStream *input);
  ~MDParser();

  virtual std::string getGrammarFileName() const override;
  virtual const antlr4::atn::ATN& getATN() const override { return _atn; };
  virtual const std::vector<std::string>& getTokenNames() const override { return _tokenNames; }; // deprecated: use vocabulary instead.
  virtual const std::vector<std::string>& getRuleNames() const override;
  virtual antlr4::dfa::Vocabulary& getVocabulary() const override;


  class ProgramContext;
  class FuncsContext;
  class FuncContext;
  class FuncDecContext;
  class GVarDefContext;
  class StmtSeqContext;
  class StmtContext;
  class ExprContext;
  class ExprsContext;
  class MayExprContext;
  class VarContext;
  class VarsContext;
  class VarDefContext;
  class VarDefsContext;
  class ArgContext;
  class ArgsContext; 

  class  ProgramContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<ProgramNode> node;
    MDParser::FuncsContext *funcsContext = nullptr;;
    ProgramContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FuncsContext *funcs();
    antlr4::tree::TerminalNode *EOF();

   
  };

  ProgramContext* program();

  class  FuncsContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<ProgramNode> node;
    MDParser::FuncsContext *part = nullptr;;
    MDParser::FuncContext *funcContext = nullptr;;
    MDParser::FuncDecContext *funcDecContext = nullptr;;
    MDParser::GVarDefContext *gVarDefContext = nullptr;;
    FuncsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    FuncContext *func();
    FuncsContext *funcs();
    FuncDecContext *funcDec();
    GVarDefContext *gVarDef();

   
  };

  FuncsContext* funcs();
  FuncsContext* funcs(int precedence);
  class  FuncContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<FunctionNode> node;
    antlr4::Token *identifierToken = nullptr;;
    MDParser::ArgsContext *argsContext = nullptr;;
    MDParser::StmtSeqContext *stmtSeqContext = nullptr;;
    FuncContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *Identifier();
    ArgsContext *args();
    StmtSeqContext *stmtSeq();

   
  };

  FuncContext* func();

  class  FuncDecContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<FunctionNode> node;
    antlr4::Token *identifierToken = nullptr;;
    MDParser::ArgsContext *argsContext = nullptr;;
    FuncDecContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *Identifier();
    ArgsContext *args();

   
  };

  FuncDecContext* funcDec();

  class  GVarDefContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<GlobalVarDefNode> node;
    antlr4::Token *identifierToken = nullptr;;
    MDParser::ExprContext *exprContext = nullptr;;
    GVarDefContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *Identifier();
    ExprContext *expr();

   
  };

  GVarDefContext* gVarDef();

  class  StmtSeqContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<StmtSeqNode> node;
    MDParser::StmtSeqContext *part = nullptr;;
    MDParser::StmtContext *stmtContext = nullptr;;
    StmtSeqContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    StmtContext *stmt();
    StmtSeqContext *stmtSeq();

   
  };

  StmtSeqContext* stmtSeq();
  StmtSeqContext* stmtSeq(int precedence);
  class  StmtContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<StmtNode> node;
    MDParser::ExprContext *exprContext = nullptr;;
    MDParser::VarDefsContext *varDefsContext = nullptr;;
    MDParser::StmtContext *stmtContext = nullptr;;
    MDParser::StmtContext *thenCase = nullptr;;
    MDParser::StmtContext *elseCase = nullptr;;
    MDParser::MayExprContext *init = nullptr;;
    MDParser::MayExprContext *cond = nullptr;;
    MDParser::MayExprContext *incr = nullptr;;
    MDParser::StmtSeqContext *stmtSeqContext = nullptr;;
    StmtContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    VarDefsContext *varDefs();
    antlr4::tree::TerminalNode *IF();
    std::vector<StmtContext *> stmt();
    StmtContext* stmt(size_t i);
    antlr4::tree::TerminalNode *ELSE();
    antlr4::tree::TerminalNode *WHILE();
    antlr4::tree::TerminalNode *DO();
    antlr4::tree::TerminalNode *FOR();
    std::vector<MayExprContext *> mayExpr();
    MayExprContext* mayExpr(size_t i);
    antlr4::tree::TerminalNode *RETURN();
    antlr4::tree::TerminalNode *BREAK();
    antlr4::tree::TerminalNode *CONTINUE();
    StmtSeqContext *stmtSeq();

   
  };

  StmtContext* stmt();

  class  ExprContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<ExprNode> node;
    MDParser::ExprContext *lhs = nullptr;;
    MDParser::ExprContext *a = nullptr;;
    antlr4::Token *integerToken = nullptr;;
    MDParser::VarContext *varContext = nullptr;;
    antlr4::Token *identifierToken = nullptr;;
    MDParser::ExprsContext *exprsContext = nullptr;;
    MDParser::ExprContext *exprContext = nullptr;;
    antlr4::Token *op = nullptr;;
    MDParser::ExprContext *rhs = nullptr;;
    MDParser::ExprContext *b = nullptr;;
    MDParser::ExprContext *c = nullptr;;
    ExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Integer();
    VarContext *var();
    antlr4::tree::TerminalNode *Identifier();
    ExprsContext *exprs();
    std::vector<ExprContext *> expr();
    ExprContext* expr(size_t i);

   
  };

  ExprContext* expr();
  ExprContext* expr(int precedence);
  class  ExprsContext : public antlr4::ParserRuleContext {
  public:
    std::vector<std::shared_ptr<ExprNode>> nodes;
    MDParser::ExprsContext *part = nullptr;;
    MDParser::ExprContext *exprContext = nullptr;;
    ExprsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();
    ExprsContext *exprs();

   
  };

  ExprsContext* exprs();
  ExprsContext* exprs(int precedence);
  class  MayExprContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<ExprNode> node;
    MDParser::ExprContext *exprContext = nullptr;;
    MayExprContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ExprContext *expr();

   
  };

  MayExprContext* mayExpr();

  class  VarContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<VarNode> node;
    antlr4::Token *identifierToken = nullptr;;
    VarContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();

   
  };

  VarContext* var();

  class  VarsContext : public antlr4::ParserRuleContext {
  public:
    std::vector<std::shared_ptr<VarNode>> nodes;
    MDParser::VarsContext *part = nullptr;;
    MDParser::VarContext *varContext = nullptr;;
    VarsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    VarContext *var();
    VarsContext *vars();

   
  };

  VarsContext* vars();
  VarsContext* vars(int precedence);
  class  VarDefContext : public antlr4::ParserRuleContext {
  public:
    std::shared_ptr<StmtNode> node;
    antlr4::Token *identifierToken = nullptr;;
    MDParser::ExprContext *exprContext = nullptr;;
    VarDefContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *Identifier();
    ExprContext *expr();

   
  };

  VarDefContext* varDef();

  class  VarDefsContext : public antlr4::ParserRuleContext {
  public:
    std::vector<std::shared_ptr<StmtNode>> nodes;
    MDParser::VarDefsContext *part = nullptr;;
    MDParser::VarDefContext *varDefContext = nullptr;;
    VarDefsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    VarDefContext *varDef();
    VarDefsContext *varDefs();

   
  };

  VarDefsContext* varDefs();
  VarDefsContext* varDefs(int precedence);
  class  ArgContext : public antlr4::ParserRuleContext {
  public:
    std::pair<ExprType, std::string> node;
    antlr4::Token *identifierToken = nullptr;;
    ArgContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    antlr4::tree::TerminalNode *INT();
    antlr4::tree::TerminalNode *Identifier();

   
  };

  ArgContext* arg();

  class  ArgsContext : public antlr4::ParserRuleContext {
  public:
    std::vector<std::pair<ExprType, std::string>> nodes;
    MDParser::ArgsContext *part = nullptr;;
    MDParser::ArgContext *argContext = nullptr;;
    ArgsContext(antlr4::ParserRuleContext *parent, size_t invokingState);
    virtual size_t getRuleIndex() const override;
    ArgContext *arg();
    ArgsContext *args();

   
  };

  ArgsContext* args();
  ArgsContext* args(int precedence);

  virtual bool sempred(antlr4::RuleContext *_localctx, size_t ruleIndex, size_t predicateIndex) override;
  bool funcsSempred(FuncsContext *_localctx, size_t predicateIndex);
  bool stmtSeqSempred(StmtSeqContext *_localctx, size_t predicateIndex);
  bool exprSempred(ExprContext *_localctx, size_t predicateIndex);
  bool exprsSempred(ExprsContext *_localctx, size_t predicateIndex);
  bool varsSempred(VarsContext *_localctx, size_t predicateIndex);
  bool varDefsSempred(VarDefsContext *_localctx, size_t predicateIndex);
  bool argsSempred(ArgsContext *_localctx, size_t predicateIndex);

private:
  static std::vector<antlr4::dfa::DFA> _decisionToDFA;
  static antlr4::atn::PredictionContextCache _sharedContextCache;
  static std::vector<std::string> _ruleNames;
  static std::vector<std::string> _tokenNames;

  static std::vector<std::string> _literalNames;
  static std::vector<std::string> _symbolicNames;
  static antlr4::dfa::Vocabulary _vocabulary;
  static antlr4::atn::ATN _atn;
  static std::vector<uint16_t> _serializedATN;


  struct Initializer {
    Initializer();
  };
  static Initializer _init;
};

