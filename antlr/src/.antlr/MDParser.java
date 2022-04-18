// Generated from /home/zhyuan020/IR/antlr/src/MDParser.g4 by ANTLR 4.8
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.misc.*;
import org.antlr.v4.runtime.tree.*;
import java.util.List;
import java.util.Iterator;
import java.util.ArrayList;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class MDParser extends Parser {
	static { RuntimeMetaData.checkVersion("4.8", RuntimeMetaData.VERSION); }

	protected static final DFA[] _decisionToDFA;
	protected static final PredictionContextCache _sharedContextCache =
		new PredictionContextCache();
	public static final int
		WhiteSpaces=1, IF=2, ELSE=3, DO=4, WHILE=5, FOR=6, RETURN=7, BREAK=8, 
		CONTINUE=9, INT=10, Integer=11, Identifier=12, ASSIGN=13, PLUS=14, MINUS=15, 
		STAR=16, SLASH=17, PERCENT=18, NOT=19, TILDE=20, AND=21, HAT=22, OR=23, 
		SL=24, SR=25, EQ=26, NE=27, LT=28, GT=29, LE=30, GE=31, LAND=32, LOR=33, 
		COLON=34, QUESTION=35, SEMICOLON=36, LPAREN=37, RPAREN=38, LBRACK=39, 
		RBRACK=40, COMMA=41;
	public static final int
		RULE_program = 0, RULE_funcs = 1, RULE_func = 2, RULE_funcDec = 3, RULE_gVarDef = 4, 
		RULE_stmtSeq = 5, RULE_stmt = 6, RULE_expr = 7, RULE_exprs = 8, RULE_mayExpr = 9, 
		RULE_var = 10, RULE_vars = 11, RULE_varDef = 12, RULE_varDefs = 13, RULE_arg = 14, 
		RULE_args = 15;
	private static String[] makeRuleNames() {
		return new String[] {
			"program", "funcs", "func", "funcDec", "gVarDef", "stmtSeq", "stmt", 
			"expr", "exprs", "mayExpr", "var", "vars", "varDef", "varDefs", "arg", 
			"args"
		};
	}
	public static final String[] ruleNames = makeRuleNames();

	private static String[] makeLiteralNames() {
		return new String[] {
			null, null, "'if'", "'else'", "'do'", "'while'", "'for'", "'return'", 
			"'break'", "'continue'", "'int'", null, null, "'='", "'+'", "'-'", "'*'", 
			"'/'", "'%'", "'!'", "'~'", "'&'", "'^'", "'|'", "'<<'", "'>>'", "'=='", 
			"'!='", "'<'", "'>'", "'<='", "'>='", "'&&'", "'||'", "':'", "'?'", "';'", 
			"'('", "')'", "'{'", "'}'", "','"
		};
	}
	private static final String[] _LITERAL_NAMES = makeLiteralNames();
	private static String[] makeSymbolicNames() {
		return new String[] {
			null, "WhiteSpaces", "IF", "ELSE", "DO", "WHILE", "FOR", "RETURN", "BREAK", 
			"CONTINUE", "INT", "Integer", "Identifier", "ASSIGN", "PLUS", "MINUS", 
			"STAR", "SLASH", "PERCENT", "NOT", "TILDE", "AND", "HAT", "OR", "SL", 
			"SR", "EQ", "NE", "LT", "GT", "LE", "GE", "LAND", "LOR", "COLON", "QUESTION", 
			"SEMICOLON", "LPAREN", "RPAREN", "LBRACK", "RBRACK", "COMMA"
		};
	}
	private static final String[] _SYMBOLIC_NAMES = makeSymbolicNames();
	public static final Vocabulary VOCABULARY = new VocabularyImpl(_LITERAL_NAMES, _SYMBOLIC_NAMES);

	/**
	 * @deprecated Use {@link #VOCABULARY} instead.
	 */
	@Deprecated
	public static final String[] tokenNames;
	static {
		tokenNames = new String[_SYMBOLIC_NAMES.length];
		for (int i = 0; i < tokenNames.length; i++) {
			tokenNames[i] = VOCABULARY.getLiteralName(i);
			if (tokenNames[i] == null) {
				tokenNames[i] = VOCABULARY.getSymbolicName(i);
			}

			if (tokenNames[i] == null) {
				tokenNames[i] = "<INVALID>";
			}
		}
	}

	@Override
	@Deprecated
	public String[] getTokenNames() {
		return tokenNames;
	}

	@Override

	public Vocabulary getVocabulary() {
		return VOCABULARY;
	}

	@Override
	public String getGrammarFileName() { return "MDParser.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public ATN getATN() { return _ATN; }

	public MDParser(TokenStream input) {
		super(input);
		_interp = new ParserATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	public static class ProgramContext extends ParserRuleContext {
		public std::shared_ptr<ProgramNode> node;
		public FuncsContext funcs;
		public FuncsContext funcs() {
			return getRuleContext(FuncsContext.class,0);
		}
		public TerminalNode EOF() { return getToken(MDParser.EOF, 0); }
		public ProgramContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_program; }
	}

	public final ProgramContext program() throws RecognitionException {
		ProgramContext _localctx = new ProgramContext(_ctx, getState());
		enterRule(_localctx, 0, RULE_program);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(32);
			((ProgramContext)_localctx).funcs = funcs(0);
			setState(33);
			match(EOF);

			            ((ProgramContext)_localctx).node =  ((ProgramContext)_localctx).funcs.node;
			          
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class FuncsContext extends ParserRuleContext {
		public std::shared_ptr<ProgramNode> node;
		public FuncsContext part;
		public FuncContext func;
		public FuncDecContext funcDec;
		public GVarDefContext gVarDef;
		public FuncContext func() {
			return getRuleContext(FuncContext.class,0);
		}
		public FuncsContext funcs() {
			return getRuleContext(FuncsContext.class,0);
		}
		public FuncDecContext funcDec() {
			return getRuleContext(FuncDecContext.class,0);
		}
		public GVarDefContext gVarDef() {
			return getRuleContext(GVarDefContext.class,0);
		}
		public FuncsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_funcs; }
	}

	public final FuncsContext funcs() throws RecognitionException {
		return funcs(0);
	}

	private FuncsContext funcs(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		FuncsContext _localctx = new FuncsContext(_ctx, _parentState);
		FuncsContext _prevctx = _localctx;
		int _startState = 2;
		enterRecursionRule(_localctx, 2, RULE_funcs, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{

			            ((FuncsContext)_localctx).node =  ProgramNode::make({});
			          
			}
			_ctx.stop = _input.LT(-1);
			setState(53);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(51);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,0,_ctx) ) {
					case 1:
						{
						_localctx = new FuncsContext(_parentctx, _parentState);
						_localctx.part = _prevctx;
						_localctx.part = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_funcs);
						setState(39);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(40);
						((FuncsContext)_localctx).func = func();

						                      ((FuncsContext)_localctx).node =  ((FuncsContext)_localctx).part.node;
						                      _localctx.node->funcs_.push_back(((FuncsContext)_localctx).func.node);
						                    
						}
						break;
					case 2:
						{
						_localctx = new FuncsContext(_parentctx, _parentState);
						_localctx.part = _prevctx;
						_localctx.part = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_funcs);
						setState(43);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(44);
						((FuncsContext)_localctx).funcDec = funcDec();

						                      ((FuncsContext)_localctx).node =  ((FuncsContext)_localctx).part.node;
						                      _localctx.node->funcs_.push_back(((FuncsContext)_localctx).funcDec.node);
						                    
						}
						break;
					case 3:
						{
						_localctx = new FuncsContext(_parentctx, _parentState);
						_localctx.part = _prevctx;
						_localctx.part = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_funcs);
						setState(47);
						if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
						setState(48);
						((FuncsContext)_localctx).gVarDef = gVarDef();

						                      ((FuncsContext)_localctx).node =  ((FuncsContext)_localctx).part.node;
						                      _localctx.node->funcs_.push_back(((FuncsContext)_localctx).gVarDef.node);
						                    
						}
						break;
					}
					} 
				}
				setState(55);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,1,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class FuncContext extends ParserRuleContext {
		public std::shared_ptr<FunctionNode> node;
		public Token Identifier;
		public ArgsContext args;
		public StmtSeqContext stmtSeq;
		public TerminalNode INT() { return getToken(MDParser.INT, 0); }
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public TerminalNode LPAREN() { return getToken(MDParser.LPAREN, 0); }
		public ArgsContext args() {
			return getRuleContext(ArgsContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(MDParser.RPAREN, 0); }
		public TerminalNode LBRACK() { return getToken(MDParser.LBRACK, 0); }
		public StmtSeqContext stmtSeq() {
			return getRuleContext(StmtSeqContext.class,0);
		}
		public TerminalNode RBRACK() { return getToken(MDParser.RBRACK, 0); }
		public FuncContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_func; }
	}

	public final FuncContext func() throws RecognitionException {
		FuncContext _localctx = new FuncContext(_ctx, getState());
		enterRule(_localctx, 4, RULE_func);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(56);
			match(INT);
			setState(57);
			((FuncContext)_localctx).Identifier = match(Identifier);
			setState(58);
			match(LPAREN);
			setState(59);
			((FuncContext)_localctx).args = args(0);
			setState(60);
			match(RPAREN);
			setState(61);
			match(LBRACK);
			setState(62);
			((FuncContext)_localctx).stmtSeq = stmtSeq(0);
			setState(63);
			match(RBRACK);

			            ((FuncContext)_localctx).node =  FunctionNode::make(ExprType::Int, (((FuncContext)_localctx).Identifier!=null?((FuncContext)_localctx).Identifier.getText():null), ((FuncContext)_localctx).args.nodes, ((FuncContext)_localctx).stmtSeq.node);
			          
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class FuncDecContext extends ParserRuleContext {
		public std::shared_ptr<FunctionNode> node;
		public Token Identifier;
		public ArgsContext args;
		public TerminalNode INT() { return getToken(MDParser.INT, 0); }
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public TerminalNode LPAREN() { return getToken(MDParser.LPAREN, 0); }
		public ArgsContext args() {
			return getRuleContext(ArgsContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(MDParser.RPAREN, 0); }
		public TerminalNode SEMICOLON() { return getToken(MDParser.SEMICOLON, 0); }
		public FuncDecContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_funcDec; }
	}

	public final FuncDecContext funcDec() throws RecognitionException {
		FuncDecContext _localctx = new FuncDecContext(_ctx, getState());
		enterRule(_localctx, 6, RULE_funcDec);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(66);
			match(INT);
			setState(67);
			((FuncDecContext)_localctx).Identifier = match(Identifier);
			setState(68);
			match(LPAREN);
			setState(69);
			((FuncDecContext)_localctx).args = args(0);
			setState(70);
			match(RPAREN);
			setState(71);
			match(SEMICOLON);

			            ((FuncDecContext)_localctx).node =  FunctionNode::make(ExprType::Int, (((FuncDecContext)_localctx).Identifier!=null?((FuncDecContext)_localctx).Identifier.getText():null), ((FuncDecContext)_localctx).args.nodes, nullptr);
			          
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class GVarDefContext extends ParserRuleContext {
		public std::shared_ptr<GlobalVarDefNode> node;
		public Token Identifier;
		public ExprContext expr;
		public TerminalNode INT() { return getToken(MDParser.INT, 0); }
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public TerminalNode SEMICOLON() { return getToken(MDParser.SEMICOLON, 0); }
		public TerminalNode ASSIGN() { return getToken(MDParser.ASSIGN, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public GVarDefContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_gVarDef; }
	}

	public final GVarDefContext gVarDef() throws RecognitionException {
		GVarDefContext _localctx = new GVarDefContext(_ctx, getState());
		enterRule(_localctx, 8, RULE_gVarDef);
		try {
			setState(85);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,2,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(74);
				match(INT);
				setState(75);
				((GVarDefContext)_localctx).Identifier = match(Identifier);
				setState(76);
				match(SEMICOLON);

				            ((GVarDefContext)_localctx).node =  GlobalVarDefNode::make(ExprType::Int, (((GVarDefContext)_localctx).Identifier!=null?((GVarDefContext)_localctx).Identifier.getText():null));
				          
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(78);
				match(INT);
				setState(79);
				((GVarDefContext)_localctx).Identifier = match(Identifier);
				setState(80);
				match(ASSIGN);
				setState(81);
				((GVarDefContext)_localctx).expr = expr(0);
				setState(82);
				match(SEMICOLON);

				            ((GVarDefContext)_localctx).node =  GlobalVarDefNode::make(ExprType::Int, (((GVarDefContext)_localctx).Identifier!=null?((GVarDefContext)_localctx).Identifier.getText():null), (((GVarDefContext)_localctx).expr!=null?_input.getText(((GVarDefContext)_localctx).expr.start,((GVarDefContext)_localctx).expr.stop):null));
				          
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class StmtSeqContext extends ParserRuleContext {
		public std::shared_ptr<StmtSeqNode> node;
		public StmtSeqContext part;
		public StmtContext stmt;
		public StmtContext stmt() {
			return getRuleContext(StmtContext.class,0);
		}
		public StmtSeqContext stmtSeq() {
			return getRuleContext(StmtSeqContext.class,0);
		}
		public StmtSeqContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_stmtSeq; }
	}

	public final StmtSeqContext stmtSeq() throws RecognitionException {
		return stmtSeq(0);
	}

	private StmtSeqContext stmtSeq(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		StmtSeqContext _localctx = new StmtSeqContext(_ctx, _parentState);
		StmtSeqContext _prevctx = _localctx;
		int _startState = 10;
		enterRecursionRule(_localctx, 10, RULE_stmtSeq, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{

			            ((StmtSeqContext)_localctx).node =  StmtSeqNode::make({}, true);
			          
			}
			_ctx.stop = _input.LT(-1);
			setState(96);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,3,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new StmtSeqContext(_parentctx, _parentState);
					_localctx.part = _prevctx;
					_localctx.part = _prevctx;
					pushNewRecursionContext(_localctx, _startState, RULE_stmtSeq);
					setState(90);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(91);
					((StmtSeqContext)_localctx).stmt = stmt();

					                      ((StmtSeqContext)_localctx).node =  ((StmtSeqContext)_localctx).part.node;
					                      _localctx.node->stmts_.push_back(((StmtSeqContext)_localctx).stmt.node);
					                    
					}
					} 
				}
				setState(98);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,3,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class StmtContext extends ParserRuleContext {
		public std::shared_ptr<StmtNode> node;
		public ExprContext expr;
		public VarDefsContext varDefs;
		public StmtContext stmt;
		public StmtContext thenCase;
		public StmtContext elseCase;
		public MayExprContext init;
		public MayExprContext cond;
		public MayExprContext incr;
		public StmtSeqContext stmtSeq;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public List<TerminalNode> SEMICOLON() { return getTokens(MDParser.SEMICOLON); }
		public TerminalNode SEMICOLON(int i) {
			return getToken(MDParser.SEMICOLON, i);
		}
		public VarDefsContext varDefs() {
			return getRuleContext(VarDefsContext.class,0);
		}
		public TerminalNode IF() { return getToken(MDParser.IF, 0); }
		public TerminalNode LPAREN() { return getToken(MDParser.LPAREN, 0); }
		public TerminalNode RPAREN() { return getToken(MDParser.RPAREN, 0); }
		public List<StmtContext> stmt() {
			return getRuleContexts(StmtContext.class);
		}
		public StmtContext stmt(int i) {
			return getRuleContext(StmtContext.class,i);
		}
		public TerminalNode ELSE() { return getToken(MDParser.ELSE, 0); }
		public TerminalNode WHILE() { return getToken(MDParser.WHILE, 0); }
		public TerminalNode DO() { return getToken(MDParser.DO, 0); }
		public TerminalNode FOR() { return getToken(MDParser.FOR, 0); }
		public List<MayExprContext> mayExpr() {
			return getRuleContexts(MayExprContext.class);
		}
		public MayExprContext mayExpr(int i) {
			return getRuleContext(MayExprContext.class,i);
		}
		public TerminalNode RETURN() { return getToken(MDParser.RETURN, 0); }
		public TerminalNode BREAK() { return getToken(MDParser.BREAK, 0); }
		public TerminalNode CONTINUE() { return getToken(MDParser.CONTINUE, 0); }
		public TerminalNode LBRACK() { return getToken(MDParser.LBRACK, 0); }
		public StmtSeqContext stmtSeq() {
			return getRuleContext(StmtSeqContext.class,0);
		}
		public TerminalNode RBRACK() { return getToken(MDParser.RBRACK, 0); }
		public StmtContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_stmt; }
	}

	public final StmtContext stmt() throws RecognitionException {
		StmtContext _localctx = new StmtContext(_ctx, getState());
		enterRule(_localctx, 12, RULE_stmt);
		try {
			setState(179);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,4,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(99);
				((StmtContext)_localctx).expr = expr(0);
				setState(100);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  InvokeNode::make(((StmtContext)_localctx).expr.node);
				          
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(103);
				((StmtContext)_localctx).varDefs = varDefs(0);
				setState(104);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  StmtSeqNode::make(((StmtContext)_localctx).varDefs.nodes);
				          
				}
				break;
			case 3:
				enterOuterAlt(_localctx, 3);
				{
				setState(107);
				match(IF);
				setState(108);
				match(LPAREN);
				setState(109);
				((StmtContext)_localctx).expr = expr(0);
				setState(110);
				match(RPAREN);
				setState(111);
				((StmtContext)_localctx).stmt = stmt();

				            ((StmtContext)_localctx).node =  IfThenElseNode::make(((StmtContext)_localctx).expr.node, ((StmtContext)_localctx).stmt.node);
				          
				}
				break;
			case 4:
				enterOuterAlt(_localctx, 4);
				{
				setState(114);
				match(IF);
				setState(115);
				match(LPAREN);
				setState(116);
				((StmtContext)_localctx).expr = expr(0);
				setState(117);
				match(RPAREN);
				setState(118);
				((StmtContext)_localctx).thenCase = stmt();
				setState(119);
				match(ELSE);
				setState(120);
				((StmtContext)_localctx).elseCase = stmt();

				            ((StmtContext)_localctx).node =  IfThenElseNode::make(((StmtContext)_localctx).expr.node, ((StmtContext)_localctx).thenCase.node, ((StmtContext)_localctx).elseCase.node);
				          
				}
				break;
			case 5:
				enterOuterAlt(_localctx, 5);
				{
				setState(123);
				match(WHILE);
				setState(124);
				match(LPAREN);
				setState(125);
				((StmtContext)_localctx).expr = expr(0);
				setState(126);
				match(RPAREN);
				setState(127);
				((StmtContext)_localctx).stmt = stmt();

				            ((StmtContext)_localctx).node =  WhileNode::make(((StmtContext)_localctx).expr.node, ((StmtContext)_localctx).stmt.node);
				          
				}
				break;
			case 6:
				enterOuterAlt(_localctx, 6);
				{
				setState(130);
				match(DO);
				setState(131);
				((StmtContext)_localctx).stmt = stmt();
				setState(132);
				match(WHILE);
				setState(133);
				match(LPAREN);
				setState(134);
				((StmtContext)_localctx).expr = expr(0);
				setState(135);
				match(RPAREN);
				setState(136);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  DoWhileNode::make(((StmtContext)_localctx).expr.node, ((StmtContext)_localctx).stmt.node);
				          
				}
				break;
			case 7:
				enterOuterAlt(_localctx, 7);
				{
				setState(139);
				match(FOR);
				setState(140);
				match(LPAREN);
				setState(141);
				((StmtContext)_localctx).init = mayExpr();
				setState(142);
				match(SEMICOLON);
				setState(143);
				((StmtContext)_localctx).cond = mayExpr();
				setState(144);
				match(SEMICOLON);
				setState(145);
				((StmtContext)_localctx).incr = mayExpr();
				setState(146);
				match(RPAREN);
				setState(147);
				((StmtContext)_localctx).stmt = stmt();

				            std::shared_ptr<StmtNode> init, incr;
				            std::shared_ptr<ExprNode> cond;
				            if (((StmtContext)_localctx).init.node) init = InvokeNode::make(((StmtContext)_localctx).init.node); else init = StmtSeqNode::make({});
				            if (((StmtContext)_localctx).cond.node) cond = ((StmtContext)_localctx).cond.node; else cond = IntegerNode::make(1);
				            if (((StmtContext)_localctx).incr.node) incr = InvokeNode::make(((StmtContext)_localctx).incr.node); else incr = StmtSeqNode::make({});
				            ((StmtContext)_localctx).node =  ForNode::make(init, cond, incr, ((StmtContext)_localctx).stmt.node);
				          
				}
				break;
			case 8:
				enterOuterAlt(_localctx, 8);
				{
				setState(150);
				match(FOR);
				setState(151);
				match(LPAREN);
				setState(152);
				((StmtContext)_localctx).varDefs = varDefs(0);
				setState(153);
				match(SEMICOLON);
				setState(154);
				((StmtContext)_localctx).cond = mayExpr();
				setState(155);
				match(SEMICOLON);
				setState(156);
				((StmtContext)_localctx).incr = mayExpr();
				setState(157);
				match(RPAREN);
				setState(158);
				((StmtContext)_localctx).stmt = stmt();

				            std::shared_ptr<StmtNode> incr;
				            std::shared_ptr<ExprNode> cond;
				            if (((StmtContext)_localctx).cond.node) cond = ((StmtContext)_localctx).cond.node; else cond = IntegerNode::make(1);
				            if (((StmtContext)_localctx).incr.node) incr = InvokeNode::make(((StmtContext)_localctx).incr.node); else incr = StmtSeqNode::make({});
				            ((StmtContext)_localctx).node =  ForNode::make(StmtSeqNode::make(((StmtContext)_localctx).varDefs.nodes), cond, incr, ((StmtContext)_localctx).stmt.node);
				            ((StmtContext)_localctx).node =  StmtSeqNode::make({_localctx.node}, true);
				          
				}
				break;
			case 9:
				enterOuterAlt(_localctx, 9);
				{
				setState(161);
				match(RETURN);
				setState(162);
				((StmtContext)_localctx).expr = expr(0);
				setState(163);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  ReturnNode::make(((StmtContext)_localctx).expr.node);
				          
				}
				break;
			case 10:
				enterOuterAlt(_localctx, 10);
				{
				setState(166);
				match(BREAK);
				setState(167);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  BreakNode::make();
				          
				}
				break;
			case 11:
				enterOuterAlt(_localctx, 11);
				{
				setState(169);
				match(CONTINUE);
				setState(170);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  ContinueNode::make();
				          
				}
				break;
			case 12:
				enterOuterAlt(_localctx, 12);
				{
				setState(172);
				match(SEMICOLON);

				            ((StmtContext)_localctx).node =  StmtSeqNode::make({});
				          
				}
				break;
			case 13:
				enterOuterAlt(_localctx, 13);
				{
				setState(174);
				match(LBRACK);
				setState(175);
				((StmtContext)_localctx).stmtSeq = stmtSeq(0);
				setState(176);
				match(RBRACK);

				            ((StmtContext)_localctx).node =  ((StmtContext)_localctx).stmtSeq.node;
				          
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ExprContext extends ParserRuleContext {
		public std::shared_ptr<ExprNode> node;
		public ExprContext lhs;
		public ExprContext a;
		public Token Integer;
		public VarContext var;
		public Token Identifier;
		public ExprsContext exprs;
		public ExprContext expr;
		public Token op;
		public ExprContext rhs;
		public ExprContext b;
		public ExprContext c;
		public TerminalNode Integer() { return getToken(MDParser.Integer, 0); }
		public VarContext var() {
			return getRuleContext(VarContext.class,0);
		}
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public TerminalNode LPAREN() { return getToken(MDParser.LPAREN, 0); }
		public ExprsContext exprs() {
			return getRuleContext(ExprsContext.class,0);
		}
		public TerminalNode RPAREN() { return getToken(MDParser.RPAREN, 0); }
		public List<ExprContext> expr() {
			return getRuleContexts(ExprContext.class);
		}
		public ExprContext expr(int i) {
			return getRuleContext(ExprContext.class,i);
		}
		public TerminalNode MINUS() { return getToken(MDParser.MINUS, 0); }
		public TerminalNode NOT() { return getToken(MDParser.NOT, 0); }
		public TerminalNode TILDE() { return getToken(MDParser.TILDE, 0); }
		public TerminalNode ASSIGN() { return getToken(MDParser.ASSIGN, 0); }
		public TerminalNode STAR() { return getToken(MDParser.STAR, 0); }
		public TerminalNode SLASH() { return getToken(MDParser.SLASH, 0); }
		public TerminalNode PERCENT() { return getToken(MDParser.PERCENT, 0); }
		public TerminalNode PLUS() { return getToken(MDParser.PLUS, 0); }
		public TerminalNode SL() { return getToken(MDParser.SL, 0); }
		public TerminalNode SR() { return getToken(MDParser.SR, 0); }
		public TerminalNode LT() { return getToken(MDParser.LT, 0); }
		public TerminalNode GT() { return getToken(MDParser.GT, 0); }
		public TerminalNode LE() { return getToken(MDParser.LE, 0); }
		public TerminalNode GE() { return getToken(MDParser.GE, 0); }
		public TerminalNode EQ() { return getToken(MDParser.EQ, 0); }
		public TerminalNode NE() { return getToken(MDParser.NE, 0); }
		public TerminalNode AND() { return getToken(MDParser.AND, 0); }
		public TerminalNode HAT() { return getToken(MDParser.HAT, 0); }
		public TerminalNode OR() { return getToken(MDParser.OR, 0); }
		public TerminalNode LAND() { return getToken(MDParser.LAND, 0); }
		public TerminalNode LOR() { return getToken(MDParser.LOR, 0); }
		public TerminalNode QUESTION() { return getToken(MDParser.QUESTION, 0); }
		public TerminalNode COLON() { return getToken(MDParser.COLON, 0); }
		public ExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_expr; }
	}

	public final ExprContext expr() throws RecognitionException {
		return expr(0);
	}

	private ExprContext expr(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprContext _localctx = new ExprContext(_ctx, _parentState);
		ExprContext _prevctx = _localctx;
		int _startState = 14;
		enterRecursionRule(_localctx, 14, RULE_expr, _p);
		int _la;
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(215);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,5,_ctx) ) {
			case 1:
				{
				setState(182);
				((ExprContext)_localctx).Integer = match(Integer);

				            ((ExprContext)_localctx).node =  IntegerNode::make(std::stoi((((ExprContext)_localctx).Integer!=null?((ExprContext)_localctx).Integer.getText():null)));
				          
				}
				break;
			case 2:
				{
				setState(184);
				((ExprContext)_localctx).var = var();

				            ((ExprContext)_localctx).node =  ((ExprContext)_localctx).var.node;
				          
				}
				break;
			case 3:
				{
				setState(187);
				((ExprContext)_localctx).Identifier = match(Identifier);
				setState(188);
				match(LPAREN);
				setState(189);
				((ExprContext)_localctx).exprs = exprs(0);
				setState(190);
				match(RPAREN);

				            ((ExprContext)_localctx).node =  CallNode::make(ExprType::Unknown, (((ExprContext)_localctx).Identifier!=null?((ExprContext)_localctx).Identifier.getText():null), ((ExprContext)_localctx).exprs.nodes);
				          
				}
				break;
			case 4:
				{
				setState(193);
				match(LPAREN);
				setState(194);
				((ExprContext)_localctx).expr = expr(0);
				setState(195);
				match(RPAREN);

				            ((ExprContext)_localctx).node =  ((ExprContext)_localctx).expr.node;
				          
				}
				break;
			case 5:
				{
				setState(198);
				match(MINUS);
				setState(199);
				((ExprContext)_localctx).expr = expr(15);

				            ((ExprContext)_localctx).node =  SubNode::make(IntegerNode::make(0), ((ExprContext)_localctx).expr.node);
				          
				}
				break;
			case 6:
				{
				setState(202);
				match(NOT);
				setState(203);
				((ExprContext)_localctx).expr = expr(14);

				            ((ExprContext)_localctx).node =  LNotNode::make(((ExprContext)_localctx).expr.node);
				          
				}
				break;
			case 7:
				{
				setState(206);
				match(TILDE);
				setState(207);
				((ExprContext)_localctx).expr = expr(13);

				            ((ExprContext)_localctx).node =  BXorNode::make(IntegerNode::make(-1), ((ExprContext)_localctx).expr.node);
				          
				}
				break;
			case 8:
				{
				setState(210);
				((ExprContext)_localctx).Identifier = match(Identifier);
				setState(211);
				match(ASSIGN);
				setState(212);
				((ExprContext)_localctx).expr = expr(1);

				            ((ExprContext)_localctx).node =  AssignNode::make((((ExprContext)_localctx).Identifier!=null?((ExprContext)_localctx).Identifier.getText():null), ((ExprContext)_localctx).expr.node);
				          
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(276);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					setState(274);
					_errHandler.sync(this);
					switch ( getInterpreter().adaptivePredict(_input,6,_ctx) ) {
					case 1:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(217);
						if (!(precpred(_ctx, 12))) throw new FailedPredicateException(this, "precpred(_ctx, 12)");
						setState(218);
						((ExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << STAR) | (1L << SLASH) | (1L << PERCENT))) != 0)) ) {
							((ExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(219);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(13);

						                      if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "*") {
						                          ((ExprContext)_localctx).node =  MulNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "/") {
						                          ((ExprContext)_localctx).node =  DivNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "%") {
						                          ((ExprContext)_localctx).node =  ModNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      }
						                    
						}
						break;
					case 2:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(222);
						if (!(precpred(_ctx, 11))) throw new FailedPredicateException(this, "precpred(_ctx, 11)");
						setState(223);
						((ExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==PLUS || _la==MINUS) ) {
							((ExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(224);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(12);

						                      if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "+") {
						                          ((ExprContext)_localctx).node =  AddNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else {
						                          ((ExprContext)_localctx).node =  SubNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      }
						                    
						}
						break;
					case 3:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(227);
						if (!(precpred(_ctx, 10))) throw new FailedPredicateException(this, "precpred(_ctx, 10)");
						setState(228);
						((ExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==SL || _la==SR) ) {
							((ExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(229);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(11);

						                      if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "<<") {
						                          ((ExprContext)_localctx).node =  SLLNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == ">>") {
						                          ((ExprContext)_localctx).node =  SRANode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      }
						                    
						}
						break;
					case 4:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(232);
						if (!(precpred(_ctx, 9))) throw new FailedPredicateException(this, "precpred(_ctx, 9)");
						setState(233);
						((ExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !((((_la) & ~0x3f) == 0 && ((1L << _la) & ((1L << LT) | (1L << GT) | (1L << LE) | (1L << GE))) != 0)) ) {
							((ExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(234);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(10);

						                      if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "<") {
						                          ((ExprContext)_localctx).node =  LTNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == ">") {
						                          ((ExprContext)_localctx).node =  GTNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "<=") {
						                          ((ExprContext)_localctx).node =  LENode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == ">=") {
						                          ((ExprContext)_localctx).node =  GENode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      }
						                    
						}
						break;
					case 5:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(237);
						if (!(precpred(_ctx, 8))) throw new FailedPredicateException(this, "precpred(_ctx, 8)");
						setState(238);
						((ExprContext)_localctx).op = _input.LT(1);
						_la = _input.LA(1);
						if ( !(_la==EQ || _la==NE) ) {
							((ExprContext)_localctx).op = (Token)_errHandler.recoverInline(this);
						}
						else {
							if ( _input.LA(1)==Token.EOF ) matchedEOF = true;
							_errHandler.reportMatch(this);
							consume();
						}
						setState(239);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(9);

						                      if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "==") {
						                          ((ExprContext)_localctx).node =  EQNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      } else if ((((ExprContext)_localctx).op!=null?((ExprContext)_localctx).op.getText():null) == "!=") {
						                          ((ExprContext)_localctx).node =  NENode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                      }
						                    
						}
						break;
					case 6:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(242);
						if (!(precpred(_ctx, 7))) throw new FailedPredicateException(this, "precpred(_ctx, 7)");
						setState(243);
						match(AND);
						setState(244);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(8);

						                      ((ExprContext)_localctx).node =  BAndNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                    
						}
						break;
					case 7:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(247);
						if (!(precpred(_ctx, 6))) throw new FailedPredicateException(this, "precpred(_ctx, 6)");
						setState(248);
						match(HAT);
						setState(249);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(7);

						                      ((ExprContext)_localctx).node =  BOrNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                    
						}
						break;
					case 8:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(252);
						if (!(precpred(_ctx, 5))) throw new FailedPredicateException(this, "precpred(_ctx, 5)");
						setState(253);
						match(OR);
						setState(254);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(6);

						                      ((ExprContext)_localctx).node =  BXorNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                    
						}
						break;
					case 9:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(257);
						if (!(precpred(_ctx, 4))) throw new FailedPredicateException(this, "precpred(_ctx, 4)");
						setState(258);
						match(LAND);
						setState(259);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(5);

						                      ((ExprContext)_localctx).node =  LAndNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                    
						}
						break;
					case 10:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.lhs = _prevctx;
						_localctx.lhs = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(262);
						if (!(precpred(_ctx, 3))) throw new FailedPredicateException(this, "precpred(_ctx, 3)");
						setState(263);
						match(LOR);
						setState(264);
						((ExprContext)_localctx).rhs = ((ExprContext)_localctx).expr = expr(4);

						                      ((ExprContext)_localctx).node =  LOrNode::make(((ExprContext)_localctx).lhs.node, ((ExprContext)_localctx).rhs.node);
						                    
						}
						break;
					case 11:
						{
						_localctx = new ExprContext(_parentctx, _parentState);
						_localctx.a = _prevctx;
						_localctx.a = _prevctx;
						pushNewRecursionContext(_localctx, _startState, RULE_expr);
						setState(267);
						if (!(precpred(_ctx, 2))) throw new FailedPredicateException(this, "precpred(_ctx, 2)");
						setState(268);
						match(QUESTION);
						setState(269);
						((ExprContext)_localctx).b = ((ExprContext)_localctx).expr = expr(0);
						setState(270);
						match(COLON);
						setState(271);
						((ExprContext)_localctx).c = ((ExprContext)_localctx).expr = expr(3);

						                      ((ExprContext)_localctx).node =  SelectNode::make(((ExprContext)_localctx).a.node, ((ExprContext)_localctx).b.node, ((ExprContext)_localctx).c.node);
						                    
						}
						break;
					}
					} 
				}
				setState(278);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,7,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ExprsContext extends ParserRuleContext {
		public std::vector<std::shared_ptr<ExprNode>> nodes;
		public ExprsContext part;
		public ExprContext expr;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public TerminalNode COMMA() { return getToken(MDParser.COMMA, 0); }
		public ExprsContext exprs() {
			return getRuleContext(ExprsContext.class,0);
		}
		public ExprsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_exprs; }
	}

	public final ExprsContext exprs() throws RecognitionException {
		return exprs(0);
	}

	private ExprsContext exprs(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ExprsContext _localctx = new ExprsContext(_ctx, _parentState);
		ExprsContext _prevctx = _localctx;
		int _startState = 16;
		enterRecursionRule(_localctx, 16, RULE_exprs, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(283);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,8,_ctx) ) {
			case 1:
				{
				}
				break;
			case 2:
				{
				setState(280);
				((ExprsContext)_localctx).expr = expr(0);

				            ((ExprsContext)_localctx).nodes =  {((ExprsContext)_localctx).expr.node};
				          
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(292);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new ExprsContext(_parentctx, _parentState);
					_localctx.part = _prevctx;
					_localctx.part = _prevctx;
					pushNewRecursionContext(_localctx, _startState, RULE_exprs);
					setState(285);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(286);
					match(COMMA);
					setState(287);
					((ExprsContext)_localctx).expr = expr(0);

					                      ((ExprsContext)_localctx).nodes =  ((ExprsContext)_localctx).part.nodes;
					                      _localctx.nodes.push_back(((ExprsContext)_localctx).expr.node);
					                    
					}
					} 
				}
				setState(294);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,9,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class MayExprContext extends ParserRuleContext {
		public std::shared_ptr<ExprNode> node;
		public ExprContext expr;
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public MayExprContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_mayExpr; }
	}

	public final MayExprContext mayExpr() throws RecognitionException {
		MayExprContext _localctx = new MayExprContext(_ctx, getState());
		enterRule(_localctx, 18, RULE_mayExpr);
		try {
			setState(299);
			_errHandler.sync(this);
			switch (_input.LA(1)) {
			case SEMICOLON:
			case RPAREN:
				enterOuterAlt(_localctx, 1);
				{
				}
				break;
			case Integer:
			case Identifier:
			case MINUS:
			case NOT:
			case TILDE:
			case LPAREN:
				enterOuterAlt(_localctx, 2);
				{
				setState(296);
				((MayExprContext)_localctx).expr = expr(0);

				            ((MayExprContext)_localctx).node =  ((MayExprContext)_localctx).expr.node;
				          
				}
				break;
			default:
				throw new NoViableAltException(this);
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarContext extends ParserRuleContext {
		public std::shared_ptr<VarNode> node;
		public Token Identifier;
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public VarContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_var; }
	}

	public final VarContext var() throws RecognitionException {
		VarContext _localctx = new VarContext(_ctx, getState());
		enterRule(_localctx, 20, RULE_var);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(301);
			((VarContext)_localctx).Identifier = match(Identifier);

			            ((VarContext)_localctx).node =  VarNode::make(ExprType::Unknown, (((VarContext)_localctx).Identifier!=null?((VarContext)_localctx).Identifier.getText():null));
			          
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarsContext extends ParserRuleContext {
		public std::vector<std::shared_ptr<VarNode>> nodes;
		public VarsContext part;
		public VarContext var;
		public VarContext var() {
			return getRuleContext(VarContext.class,0);
		}
		public TerminalNode COMMA() { return getToken(MDParser.COMMA, 0); }
		public VarsContext vars() {
			return getRuleContext(VarsContext.class,0);
		}
		public VarsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_vars; }
	}

	public final VarsContext vars() throws RecognitionException {
		return vars(0);
	}

	private VarsContext vars(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		VarsContext _localctx = new VarsContext(_ctx, _parentState);
		VarsContext _prevctx = _localctx;
		int _startState = 22;
		enterRecursionRule(_localctx, 22, RULE_vars, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(308);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,11,_ctx) ) {
			case 1:
				{
				}
				break;
			case 2:
				{
				setState(305);
				((VarsContext)_localctx).var = var();

				            ((VarsContext)_localctx).nodes =  {((VarsContext)_localctx).var.node};
				          
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(317);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new VarsContext(_parentctx, _parentState);
					_localctx.part = _prevctx;
					_localctx.part = _prevctx;
					pushNewRecursionContext(_localctx, _startState, RULE_vars);
					setState(310);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(311);
					match(COMMA);
					setState(312);
					((VarsContext)_localctx).var = var();

					                      ((VarsContext)_localctx).nodes =  ((VarsContext)_localctx).part.nodes;
					                      _localctx.nodes.push_back(((VarsContext)_localctx).var.node);
					                    
					}
					} 
				}
				setState(319);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,12,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class VarDefContext extends ParserRuleContext {
		public std::shared_ptr<StmtNode> node;
		public Token Identifier;
		public ExprContext expr;
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public TerminalNode ASSIGN() { return getToken(MDParser.ASSIGN, 0); }
		public ExprContext expr() {
			return getRuleContext(ExprContext.class,0);
		}
		public VarDefContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_varDef; }
	}

	public final VarDefContext varDef() throws RecognitionException {
		VarDefContext _localctx = new VarDefContext(_ctx, getState());
		enterRule(_localctx, 24, RULE_varDef);
		try {
			setState(327);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,13,_ctx) ) {
			case 1:
				enterOuterAlt(_localctx, 1);
				{
				setState(320);
				((VarDefContext)_localctx).Identifier = match(Identifier);

				            ((VarDefContext)_localctx).node =  VarDefNode::make(ExprType::Int, (((VarDefContext)_localctx).Identifier!=null?((VarDefContext)_localctx).Identifier.getText():null));
				          
				}
				break;
			case 2:
				enterOuterAlt(_localctx, 2);
				{
				setState(322);
				((VarDefContext)_localctx).Identifier = match(Identifier);
				setState(323);
				match(ASSIGN);
				setState(324);
				((VarDefContext)_localctx).expr = expr(0);

				            ((VarDefContext)_localctx).node =  StmtSeqNode::make({
				                        VarDefNode::make(ExprType::Int, (((VarDefContext)_localctx).Identifier!=null?((VarDefContext)_localctx).Identifier.getText():null)),
				                        InvokeNode::make(AssignNode::make((((VarDefContext)_localctx).Identifier!=null?((VarDefContext)_localctx).Identifier.getText():null), ((VarDefContext)_localctx).expr.node))});
				          
				}
				break;
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class VarDefsContext extends ParserRuleContext {
		public std::vector<std::shared_ptr<StmtNode>> nodes;
		public VarDefsContext part;
		public VarDefContext varDef;
		public TerminalNode INT() { return getToken(MDParser.INT, 0); }
		public VarDefContext varDef() {
			return getRuleContext(VarDefContext.class,0);
		}
		public TerminalNode COMMA() { return getToken(MDParser.COMMA, 0); }
		public VarDefsContext varDefs() {
			return getRuleContext(VarDefsContext.class,0);
		}
		public VarDefsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_varDefs; }
	}

	public final VarDefsContext varDefs() throws RecognitionException {
		return varDefs(0);
	}

	private VarDefsContext varDefs(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		VarDefsContext _localctx = new VarDefsContext(_ctx, _parentState);
		VarDefsContext _prevctx = _localctx;
		int _startState = 26;
		enterRecursionRule(_localctx, 26, RULE_varDefs, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			{
			setState(330);
			match(INT);
			setState(331);
			((VarDefsContext)_localctx).varDef = varDef();

			            ((VarDefsContext)_localctx).nodes =  {((VarDefsContext)_localctx).varDef.node};
			          
			}
			_ctx.stop = _input.LT(-1);
			setState(341);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,14,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new VarDefsContext(_parentctx, _parentState);
					_localctx.part = _prevctx;
					_localctx.part = _prevctx;
					pushNewRecursionContext(_localctx, _startState, RULE_varDefs);
					setState(334);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(335);
					match(COMMA);
					setState(336);
					((VarDefsContext)_localctx).varDef = varDef();

					                      ((VarDefsContext)_localctx).nodes =  ((VarDefsContext)_localctx).part.nodes;
					                      _localctx.nodes.push_back(((VarDefsContext)_localctx).varDef.node);
					                    
					}
					} 
				}
				setState(343);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,14,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public static class ArgContext extends ParserRuleContext {
		public std::pair<ExprType, std::string> node;
		public Token Identifier;
		public TerminalNode INT() { return getToken(MDParser.INT, 0); }
		public TerminalNode Identifier() { return getToken(MDParser.Identifier, 0); }
		public ArgContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_arg; }
	}

	public final ArgContext arg() throws RecognitionException {
		ArgContext _localctx = new ArgContext(_ctx, getState());
		enterRule(_localctx, 28, RULE_arg);
		try {
			enterOuterAlt(_localctx, 1);
			{
			setState(344);
			match(INT);
			setState(345);
			((ArgContext)_localctx).Identifier = match(Identifier);

			            ((ArgContext)_localctx).node =  std::make_pair(ExprType::Int, (((ArgContext)_localctx).Identifier!=null?((ArgContext)_localctx).Identifier.getText():null));
			          
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			exitRule();
		}
		return _localctx;
	}

	public static class ArgsContext extends ParserRuleContext {
		public std::vector<std::pair<ExprType, std::string>> nodes;
		public ArgsContext part;
		public ArgContext arg;
		public ArgContext arg() {
			return getRuleContext(ArgContext.class,0);
		}
		public TerminalNode COMMA() { return getToken(MDParser.COMMA, 0); }
		public ArgsContext args() {
			return getRuleContext(ArgsContext.class,0);
		}
		public ArgsContext(ParserRuleContext parent, int invokingState) {
			super(parent, invokingState);
		}
		@Override public int getRuleIndex() { return RULE_args; }
	}

	public final ArgsContext args() throws RecognitionException {
		return args(0);
	}

	private ArgsContext args(int _p) throws RecognitionException {
		ParserRuleContext _parentctx = _ctx;
		int _parentState = getState();
		ArgsContext _localctx = new ArgsContext(_ctx, _parentState);
		ArgsContext _prevctx = _localctx;
		int _startState = 30;
		enterRecursionRule(_localctx, 30, RULE_args, _p);
		try {
			int _alt;
			enterOuterAlt(_localctx, 1);
			{
			setState(352);
			_errHandler.sync(this);
			switch ( getInterpreter().adaptivePredict(_input,15,_ctx) ) {
			case 1:
				{
				}
				break;
			case 2:
				{
				setState(349);
				((ArgsContext)_localctx).arg = arg();

				            ((ArgsContext)_localctx).nodes =  {((ArgsContext)_localctx).arg.node};
				          
				}
				break;
			}
			_ctx.stop = _input.LT(-1);
			setState(361);
			_errHandler.sync(this);
			_alt = getInterpreter().adaptivePredict(_input,16,_ctx);
			while ( _alt!=2 && _alt!=org.antlr.v4.runtime.atn.ATN.INVALID_ALT_NUMBER ) {
				if ( _alt==1 ) {
					if ( _parseListeners!=null ) triggerExitRuleEvent();
					_prevctx = _localctx;
					{
					{
					_localctx = new ArgsContext(_parentctx, _parentState);
					_localctx.part = _prevctx;
					_localctx.part = _prevctx;
					pushNewRecursionContext(_localctx, _startState, RULE_args);
					setState(354);
					if (!(precpred(_ctx, 1))) throw new FailedPredicateException(this, "precpred(_ctx, 1)");
					setState(355);
					match(COMMA);
					setState(356);
					((ArgsContext)_localctx).arg = arg();

					                      ((ArgsContext)_localctx).nodes =  ((ArgsContext)_localctx).part.nodes;
					                      _localctx.nodes.push_back(((ArgsContext)_localctx).arg.node);
					                    
					}
					} 
				}
				setState(363);
				_errHandler.sync(this);
				_alt = getInterpreter().adaptivePredict(_input,16,_ctx);
			}
			}
		}
		catch (RecognitionException re) {
			_localctx.exception = re;
			_errHandler.reportError(this, re);
			_errHandler.recover(this, re);
		}
		finally {
			unrollRecursionContexts(_parentctx);
		}
		return _localctx;
	}

	public boolean sempred(RuleContext _localctx, int ruleIndex, int predIndex) {
		switch (ruleIndex) {
		case 1:
			return funcs_sempred((FuncsContext)_localctx, predIndex);
		case 5:
			return stmtSeq_sempred((StmtSeqContext)_localctx, predIndex);
		case 7:
			return expr_sempred((ExprContext)_localctx, predIndex);
		case 8:
			return exprs_sempred((ExprsContext)_localctx, predIndex);
		case 11:
			return vars_sempred((VarsContext)_localctx, predIndex);
		case 13:
			return varDefs_sempred((VarDefsContext)_localctx, predIndex);
		case 15:
			return args_sempred((ArgsContext)_localctx, predIndex);
		}
		return true;
	}
	private boolean funcs_sempred(FuncsContext _localctx, int predIndex) {
		switch (predIndex) {
		case 0:
			return precpred(_ctx, 3);
		case 1:
			return precpred(_ctx, 2);
		case 2:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean stmtSeq_sempred(StmtSeqContext _localctx, int predIndex) {
		switch (predIndex) {
		case 3:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean expr_sempred(ExprContext _localctx, int predIndex) {
		switch (predIndex) {
		case 4:
			return precpred(_ctx, 12);
		case 5:
			return precpred(_ctx, 11);
		case 6:
			return precpred(_ctx, 10);
		case 7:
			return precpred(_ctx, 9);
		case 8:
			return precpred(_ctx, 8);
		case 9:
			return precpred(_ctx, 7);
		case 10:
			return precpred(_ctx, 6);
		case 11:
			return precpred(_ctx, 5);
		case 12:
			return precpred(_ctx, 4);
		case 13:
			return precpred(_ctx, 3);
		case 14:
			return precpred(_ctx, 2);
		}
		return true;
	}
	private boolean exprs_sempred(ExprsContext _localctx, int predIndex) {
		switch (predIndex) {
		case 15:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean vars_sempred(VarsContext _localctx, int predIndex) {
		switch (predIndex) {
		case 16:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean varDefs_sempred(VarDefsContext _localctx, int predIndex) {
		switch (predIndex) {
		case 17:
			return precpred(_ctx, 1);
		}
		return true;
	}
	private boolean args_sempred(ArgsContext _localctx, int predIndex) {
		switch (predIndex) {
		case 18:
			return precpred(_ctx, 1);
		}
		return true;
	}

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3+\u016f\4\2\t\2\4"+
		"\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t"+
		"\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\3\2\3\2\3"+
		"\2\3\2\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\7\3"+
		"\66\n\3\f\3\16\39\13\3\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\5\3\5"+
		"\3\5\3\5\3\5\3\5\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\5"+
		"\6X\n\6\3\7\3\7\3\7\3\7\3\7\3\7\3\7\7\7a\n\7\f\7\16\7d\13\7\3\b\3\b\3"+
		"\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b"+
		"\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3"+
		"\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b"+
		"\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3"+
		"\b\3\b\3\b\3\b\3\b\3\b\3\b\3\b\5\b\u00b6\n\b\3\t\3\t\3\t\3\t\3\t\3\t\3"+
		"\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t"+
		"\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\5\t\u00da\n\t\3\t\3\t\3\t\3\t"+
		"\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3"+
		"\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t"+
		"\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3"+
		"\t\7\t\u0115\n\t\f\t\16\t\u0118\13\t\3\n\3\n\3\n\3\n\5\n\u011e\n\n\3\n"+
		"\3\n\3\n\3\n\3\n\7\n\u0125\n\n\f\n\16\n\u0128\13\n\3\13\3\13\3\13\3\13"+
		"\5\13\u012e\n\13\3\f\3\f\3\f\3\r\3\r\3\r\3\r\5\r\u0137\n\r\3\r\3\r\3\r"+
		"\3\r\3\r\7\r\u013e\n\r\f\r\16\r\u0141\13\r\3\16\3\16\3\16\3\16\3\16\3"+
		"\16\3\16\5\16\u014a\n\16\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17\3\17"+
		"\3\17\7\17\u0156\n\17\f\17\16\17\u0159\13\17\3\20\3\20\3\20\3\20\3\21"+
		"\3\21\3\21\3\21\5\21\u0163\n\21\3\21\3\21\3\21\3\21\3\21\7\21\u016a\n"+
		"\21\f\21\16\21\u016d\13\21\3\21\2\t\4\f\20\22\30\34 \22\2\4\6\b\n\f\16"+
		"\20\22\24\26\30\32\34\36 \2\7\3\2\22\24\3\2\20\21\3\2\32\33\3\2\36!\3"+
		"\2\34\35\2\u018a\2\"\3\2\2\2\4&\3\2\2\2\6:\3\2\2\2\bD\3\2\2\2\nW\3\2\2"+
		"\2\fY\3\2\2\2\16\u00b5\3\2\2\2\20\u00d9\3\2\2\2\22\u011d\3\2\2\2\24\u012d"+
		"\3\2\2\2\26\u012f\3\2\2\2\30\u0136\3\2\2\2\32\u0149\3\2\2\2\34\u014b\3"+
		"\2\2\2\36\u015a\3\2\2\2 \u0162\3\2\2\2\"#\5\4\3\2#$\7\2\2\3$%\b\2\1\2"+
		"%\3\3\2\2\2&\'\b\3\1\2\'(\b\3\1\2(\67\3\2\2\2)*\f\5\2\2*+\5\6\4\2+,\b"+
		"\3\1\2,\66\3\2\2\2-.\f\4\2\2./\5\b\5\2/\60\b\3\1\2\60\66\3\2\2\2\61\62"+
		"\f\3\2\2\62\63\5\n\6\2\63\64\b\3\1\2\64\66\3\2\2\2\65)\3\2\2\2\65-\3\2"+
		"\2\2\65\61\3\2\2\2\669\3\2\2\2\67\65\3\2\2\2\678\3\2\2\28\5\3\2\2\29\67"+
		"\3\2\2\2:;\7\f\2\2;<\7\16\2\2<=\7\'\2\2=>\5 \21\2>?\7(\2\2?@\7)\2\2@A"+
		"\5\f\7\2AB\7*\2\2BC\b\4\1\2C\7\3\2\2\2DE\7\f\2\2EF\7\16\2\2FG\7\'\2\2"+
		"GH\5 \21\2HI\7(\2\2IJ\7&\2\2JK\b\5\1\2K\t\3\2\2\2LM\7\f\2\2MN\7\16\2\2"+
		"NO\7&\2\2OX\b\6\1\2PQ\7\f\2\2QR\7\16\2\2RS\7\17\2\2ST\5\20\t\2TU\7&\2"+
		"\2UV\b\6\1\2VX\3\2\2\2WL\3\2\2\2WP\3\2\2\2X\13\3\2\2\2YZ\b\7\1\2Z[\b\7"+
		"\1\2[b\3\2\2\2\\]\f\3\2\2]^\5\16\b\2^_\b\7\1\2_a\3\2\2\2`\\\3\2\2\2ad"+
		"\3\2\2\2b`\3\2\2\2bc\3\2\2\2c\r\3\2\2\2db\3\2\2\2ef\5\20\t\2fg\7&\2\2"+
		"gh\b\b\1\2h\u00b6\3\2\2\2ij\5\34\17\2jk\7&\2\2kl\b\b\1\2l\u00b6\3\2\2"+
		"\2mn\7\4\2\2no\7\'\2\2op\5\20\t\2pq\7(\2\2qr\5\16\b\2rs\b\b\1\2s\u00b6"+
		"\3\2\2\2tu\7\4\2\2uv\7\'\2\2vw\5\20\t\2wx\7(\2\2xy\5\16\b\2yz\7\5\2\2"+
		"z{\5\16\b\2{|\b\b\1\2|\u00b6\3\2\2\2}~\7\7\2\2~\177\7\'\2\2\177\u0080"+
		"\5\20\t\2\u0080\u0081\7(\2\2\u0081\u0082\5\16\b\2\u0082\u0083\b\b\1\2"+
		"\u0083\u00b6\3\2\2\2\u0084\u0085\7\6\2\2\u0085\u0086\5\16\b\2\u0086\u0087"+
		"\7\7\2\2\u0087\u0088\7\'\2\2\u0088\u0089\5\20\t\2\u0089\u008a\7(\2\2\u008a"+
		"\u008b\7&\2\2\u008b\u008c\b\b\1\2\u008c\u00b6\3\2\2\2\u008d\u008e\7\b"+
		"\2\2\u008e\u008f\7\'\2\2\u008f\u0090\5\24\13\2\u0090\u0091\7&\2\2\u0091"+
		"\u0092\5\24\13\2\u0092\u0093\7&\2\2\u0093\u0094\5\24\13\2\u0094\u0095"+
		"\7(\2\2\u0095\u0096\5\16\b\2\u0096\u0097\b\b\1\2\u0097\u00b6\3\2\2\2\u0098"+
		"\u0099\7\b\2\2\u0099\u009a\7\'\2\2\u009a\u009b\5\34\17\2\u009b\u009c\7"+
		"&\2\2\u009c\u009d\5\24\13\2\u009d\u009e\7&\2\2\u009e\u009f\5\24\13\2\u009f"+
		"\u00a0\7(\2\2\u00a0\u00a1\5\16\b\2\u00a1\u00a2\b\b\1\2\u00a2\u00b6\3\2"+
		"\2\2\u00a3\u00a4\7\t\2\2\u00a4\u00a5\5\20\t\2\u00a5\u00a6\7&\2\2\u00a6"+
		"\u00a7\b\b\1\2\u00a7\u00b6\3\2\2\2\u00a8\u00a9\7\n\2\2\u00a9\u00aa\7&"+
		"\2\2\u00aa\u00b6\b\b\1\2\u00ab\u00ac\7\13\2\2\u00ac\u00ad\7&\2\2\u00ad"+
		"\u00b6\b\b\1\2\u00ae\u00af\7&\2\2\u00af\u00b6\b\b\1\2\u00b0\u00b1\7)\2"+
		"\2\u00b1\u00b2\5\f\7\2\u00b2\u00b3\7*\2\2\u00b3\u00b4\b\b\1\2\u00b4\u00b6"+
		"\3\2\2\2\u00b5e\3\2\2\2\u00b5i\3\2\2\2\u00b5m\3\2\2\2\u00b5t\3\2\2\2\u00b5"+
		"}\3\2\2\2\u00b5\u0084\3\2\2\2\u00b5\u008d\3\2\2\2\u00b5\u0098\3\2\2\2"+
		"\u00b5\u00a3\3\2\2\2\u00b5\u00a8\3\2\2\2\u00b5\u00ab\3\2\2\2\u00b5\u00ae"+
		"\3\2\2\2\u00b5\u00b0\3\2\2\2\u00b6\17\3\2\2\2\u00b7\u00b8\b\t\1\2\u00b8"+
		"\u00b9\7\r\2\2\u00b9\u00da\b\t\1\2\u00ba\u00bb\5\26\f\2\u00bb\u00bc\b"+
		"\t\1\2\u00bc\u00da\3\2\2\2\u00bd\u00be\7\16\2\2\u00be\u00bf\7\'\2\2\u00bf"+
		"\u00c0\5\22\n\2\u00c0\u00c1\7(\2\2\u00c1\u00c2\b\t\1\2\u00c2\u00da\3\2"+
		"\2\2\u00c3\u00c4\7\'\2\2\u00c4\u00c5\5\20\t\2\u00c5\u00c6\7(\2\2\u00c6"+
		"\u00c7\b\t\1\2\u00c7\u00da\3\2\2\2\u00c8\u00c9\7\21\2\2\u00c9\u00ca\5"+
		"\20\t\21\u00ca\u00cb\b\t\1\2\u00cb\u00da\3\2\2\2\u00cc\u00cd\7\25\2\2"+
		"\u00cd\u00ce\5\20\t\20\u00ce\u00cf\b\t\1\2\u00cf\u00da\3\2\2\2\u00d0\u00d1"+
		"\7\26\2\2\u00d1\u00d2\5\20\t\17\u00d2\u00d3\b\t\1\2\u00d3\u00da\3\2\2"+
		"\2\u00d4\u00d5\7\16\2\2\u00d5\u00d6\7\17\2\2\u00d6\u00d7\5\20\t\3\u00d7"+
		"\u00d8\b\t\1\2\u00d8\u00da\3\2\2\2\u00d9\u00b7\3\2\2\2\u00d9\u00ba\3\2"+
		"\2\2\u00d9\u00bd\3\2\2\2\u00d9\u00c3\3\2\2\2\u00d9\u00c8\3\2\2\2\u00d9"+
		"\u00cc\3\2\2\2\u00d9\u00d0\3\2\2\2\u00d9\u00d4\3\2\2\2\u00da\u0116\3\2"+
		"\2\2\u00db\u00dc\f\16\2\2\u00dc\u00dd\t\2\2\2\u00dd\u00de\5\20\t\17\u00de"+
		"\u00df\b\t\1\2\u00df\u0115\3\2\2\2\u00e0\u00e1\f\r\2\2\u00e1\u00e2\t\3"+
		"\2\2\u00e2\u00e3\5\20\t\16\u00e3\u00e4\b\t\1\2\u00e4\u0115\3\2\2\2\u00e5"+
		"\u00e6\f\f\2\2\u00e6\u00e7\t\4\2\2\u00e7\u00e8\5\20\t\r\u00e8\u00e9\b"+
		"\t\1\2\u00e9\u0115\3\2\2\2\u00ea\u00eb\f\13\2\2\u00eb\u00ec\t\5\2\2\u00ec"+
		"\u00ed\5\20\t\f\u00ed\u00ee\b\t\1\2\u00ee\u0115\3\2\2\2\u00ef\u00f0\f"+
		"\n\2\2\u00f0\u00f1\t\6\2\2\u00f1\u00f2\5\20\t\13\u00f2\u00f3\b\t\1\2\u00f3"+
		"\u0115\3\2\2\2\u00f4\u00f5\f\t\2\2\u00f5\u00f6\7\27\2\2\u00f6\u00f7\5"+
		"\20\t\n\u00f7\u00f8\b\t\1\2\u00f8\u0115\3\2\2\2\u00f9\u00fa\f\b\2\2\u00fa"+
		"\u00fb\7\30\2\2\u00fb\u00fc\5\20\t\t\u00fc\u00fd\b\t\1\2\u00fd\u0115\3"+
		"\2\2\2\u00fe\u00ff\f\7\2\2\u00ff\u0100\7\31\2\2\u0100\u0101\5\20\t\b\u0101"+
		"\u0102\b\t\1\2\u0102\u0115\3\2\2\2\u0103\u0104\f\6\2\2\u0104\u0105\7\""+
		"\2\2\u0105\u0106\5\20\t\7\u0106\u0107\b\t\1\2\u0107\u0115\3\2\2\2\u0108"+
		"\u0109\f\5\2\2\u0109\u010a\7#\2\2\u010a\u010b\5\20\t\6\u010b\u010c\b\t"+
		"\1\2\u010c\u0115\3\2\2\2\u010d\u010e\f\4\2\2\u010e\u010f\7%\2\2\u010f"+
		"\u0110\5\20\t\2\u0110\u0111\7$\2\2\u0111\u0112\5\20\t\5\u0112\u0113\b"+
		"\t\1\2\u0113\u0115\3\2\2\2\u0114\u00db\3\2\2\2\u0114\u00e0\3\2\2\2\u0114"+
		"\u00e5\3\2\2\2\u0114\u00ea\3\2\2\2\u0114\u00ef\3\2\2\2\u0114\u00f4\3\2"+
		"\2\2\u0114\u00f9\3\2\2\2\u0114\u00fe\3\2\2\2\u0114\u0103\3\2\2\2\u0114"+
		"\u0108\3\2\2\2\u0114\u010d\3\2\2\2\u0115\u0118\3\2\2\2\u0116\u0114\3\2"+
		"\2\2\u0116\u0117\3\2\2\2\u0117\21\3\2\2\2\u0118\u0116\3\2\2\2\u0119\u011e"+
		"\b\n\1\2\u011a\u011b\5\20\t\2\u011b\u011c\b\n\1\2\u011c\u011e\3\2\2\2"+
		"\u011d\u0119\3\2\2\2\u011d\u011a\3\2\2\2\u011e\u0126\3\2\2\2\u011f\u0120"+
		"\f\3\2\2\u0120\u0121\7+\2\2\u0121\u0122\5\20\t\2\u0122\u0123\b\n\1\2\u0123"+
		"\u0125\3\2\2\2\u0124\u011f\3\2\2\2\u0125\u0128\3\2\2\2\u0126\u0124\3\2"+
		"\2\2\u0126\u0127\3\2\2\2\u0127\23\3\2\2\2\u0128\u0126\3\2\2\2\u0129\u012e"+
		"\3\2\2\2\u012a\u012b\5\20\t\2\u012b\u012c\b\13\1\2\u012c\u012e\3\2\2\2"+
		"\u012d\u0129\3\2\2\2\u012d\u012a\3\2\2\2\u012e\25\3\2\2\2\u012f\u0130"+
		"\7\16\2\2\u0130\u0131\b\f\1\2\u0131\27\3\2\2\2\u0132\u0137\b\r\1\2\u0133"+
		"\u0134\5\26\f\2\u0134\u0135\b\r\1\2\u0135\u0137\3\2\2\2\u0136\u0132\3"+
		"\2\2\2\u0136\u0133\3\2\2\2\u0137\u013f\3\2\2\2\u0138\u0139\f\3\2\2\u0139"+
		"\u013a\7+\2\2\u013a\u013b\5\26\f\2\u013b\u013c\b\r\1\2\u013c\u013e\3\2"+
		"\2\2\u013d\u0138\3\2\2\2\u013e\u0141\3\2\2\2\u013f\u013d\3\2\2\2\u013f"+
		"\u0140\3\2\2\2\u0140\31\3\2\2\2\u0141\u013f\3\2\2\2\u0142\u0143\7\16\2"+
		"\2\u0143\u014a\b\16\1\2\u0144\u0145\7\16\2\2\u0145\u0146\7\17\2\2\u0146"+
		"\u0147\5\20\t\2\u0147\u0148\b\16\1\2\u0148\u014a\3\2\2\2\u0149\u0142\3"+
		"\2\2\2\u0149\u0144\3\2\2\2\u014a\33\3\2\2\2\u014b\u014c\b\17\1\2\u014c"+
		"\u014d\7\f\2\2\u014d\u014e\5\32\16\2\u014e\u014f\b\17\1\2\u014f\u0157"+
		"\3\2\2\2\u0150\u0151\f\3\2\2\u0151\u0152\7+\2\2\u0152\u0153\5\32\16\2"+
		"\u0153\u0154\b\17\1\2\u0154\u0156\3\2\2\2\u0155\u0150\3\2\2\2\u0156\u0159"+
		"\3\2\2\2\u0157\u0155\3\2\2\2\u0157\u0158\3\2\2\2\u0158\35\3\2\2\2\u0159"+
		"\u0157\3\2\2\2\u015a\u015b\7\f\2\2\u015b\u015c\7\16\2\2\u015c\u015d\b"+
		"\20\1\2\u015d\37\3\2\2\2\u015e\u0163\b\21\1\2\u015f\u0160\5\36\20\2\u0160"+
		"\u0161\b\21\1\2\u0161\u0163\3\2\2\2\u0162\u015e\3\2\2\2\u0162\u015f\3"+
		"\2\2\2\u0163\u016b\3\2\2\2\u0164\u0165\f\3\2\2\u0165\u0166\7+\2\2\u0166"+
		"\u0167\5\36\20\2\u0167\u0168\b\21\1\2\u0168\u016a\3\2\2\2\u0169\u0164"+
		"\3\2\2\2\u016a\u016d\3\2\2\2\u016b\u0169\3\2\2\2\u016b\u016c\3\2\2\2\u016c"+
		"!\3\2\2\2\u016d\u016b\3\2\2\2\23\65\67Wb\u00b5\u00d9\u0114\u0116\u011d"+
		"\u0126\u012d\u0136\u013f\u0149\u0157\u0162\u016b";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}