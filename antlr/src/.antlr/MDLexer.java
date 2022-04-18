// Generated from /home/zhyuan020/IR/antlr/src/MDLexer.g4 by ANTLR 4.8
import org.antlr.v4.runtime.Lexer;
import org.antlr.v4.runtime.CharStream;
import org.antlr.v4.runtime.Token;
import org.antlr.v4.runtime.TokenStream;
import org.antlr.v4.runtime.*;
import org.antlr.v4.runtime.atn.*;
import org.antlr.v4.runtime.dfa.DFA;
import org.antlr.v4.runtime.misc.*;

@SuppressWarnings({"all", "warnings", "unchecked", "unused", "cast"})
public class MDLexer extends Lexer {
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
	public static String[] channelNames = {
		"DEFAULT_TOKEN_CHANNEL", "HIDDEN"
	};

	public static String[] modeNames = {
		"DEFAULT_MODE"
	};

	private static String[] makeRuleNames() {
		return new String[] {
			"WhiteSpaces", "IF", "ELSE", "DO", "WHILE", "FOR", "RETURN", "BREAK", 
			"CONTINUE", "INT", "Integer", "Identifier", "ASSIGN", "PLUS", "MINUS", 
			"STAR", "SLASH", "PERCENT", "NOT", "TILDE", "AND", "HAT", "OR", "SL", 
			"SR", "EQ", "NE", "LT", "GT", "LE", "GE", "LAND", "LOR", "COLON", "QUESTION", 
			"SEMICOLON", "LPAREN", "RPAREN", "LBRACK", "RBRACK", "COMMA"
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


	public MDLexer(CharStream input) {
		super(input);
		_interp = new LexerATNSimulator(this,_ATN,_decisionToDFA,_sharedContextCache);
	}

	@Override
	public String getGrammarFileName() { return "MDLexer.g4"; }

	@Override
	public String[] getRuleNames() { return ruleNames; }

	@Override
	public String getSerializedATN() { return _serializedATN; }

	@Override
	public String[] getChannelNames() { return channelNames; }

	@Override
	public String[] getModeNames() { return modeNames; }

	@Override
	public ATN getATN() { return _ATN; }

	public static final String _serializedATN =
		"\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2+\u00d9\b\1\4\2\t"+
		"\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13"+
		"\t\13\4\f\t\f\4\r\t\r\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22"+
		"\4\23\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31\t\31"+
		"\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36\t\36\4\37\t\37\4 \t \4!"+
		"\t!\4\"\t\"\4#\t#\4$\t$\4%\t%\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\3\2\6\2"+
		"W\n\2\r\2\16\2X\3\2\3\2\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4\3\5\3\5\3\5\3"+
		"\6\3\6\3\6\3\6\3\6\3\6\3\7\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3\b\3\b\3\b\3\t"+
		"\3\t\3\t\3\t\3\t\3\t\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\13\3\13\3\13"+
		"\3\13\3\f\6\f\u008d\n\f\r\f\16\f\u008e\3\r\3\r\7\r\u0093\n\r\f\r\16\r"+
		"\u0096\13\r\3\16\3\16\3\17\3\17\3\20\3\20\3\21\3\21\3\22\3\22\3\23\3\23"+
		"\3\24\3\24\3\25\3\25\3\26\3\26\3\27\3\27\3\30\3\30\3\31\3\31\3\31\3\32"+
		"\3\32\3\32\3\33\3\33\3\33\3\34\3\34\3\34\3\35\3\35\3\36\3\36\3\37\3\37"+
		"\3\37\3 \3 \3 \3!\3!\3!\3\"\3\"\3\"\3#\3#\3$\3$\3%\3%\3&\3&\3\'\3\'\3"+
		"(\3(\3)\3)\3*\3*\2\2+\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27"+
		"\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30/\31\61\32\63\33"+
		"\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'M(O)Q*S+\3\2\6\5\2\13\f\17\17\""+
		"\"\3\2\62;\5\2C\\aac|\6\2\62;C\\aac|\2\u00db\2\3\3\2\2\2\2\5\3\2\2\2\2"+
		"\7\3\2\2\2\2\t\3\2\2\2\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2"+
		"\2\2\2\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33\3\2\2\2"+
		"\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2\2\2%\3\2\2\2\2\'\3\2\2"+
		"\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2"+
		"\2\2\65\3\2\2\2\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2\2"+
		"\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3\2\2\2\2K\3\2\2\2\2"+
		"M\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2S\3\2\2\2\3V\3\2\2\2\5\\\3\2\2\2\7_\3"+
		"\2\2\2\td\3\2\2\2\13g\3\2\2\2\rm\3\2\2\2\17q\3\2\2\2\21x\3\2\2\2\23~\3"+
		"\2\2\2\25\u0087\3\2\2\2\27\u008c\3\2\2\2\31\u0090\3\2\2\2\33\u0097\3\2"+
		"\2\2\35\u0099\3\2\2\2\37\u009b\3\2\2\2!\u009d\3\2\2\2#\u009f\3\2\2\2%"+
		"\u00a1\3\2\2\2\'\u00a3\3\2\2\2)\u00a5\3\2\2\2+\u00a7\3\2\2\2-\u00a9\3"+
		"\2\2\2/\u00ab\3\2\2\2\61\u00ad\3\2\2\2\63\u00b0\3\2\2\2\65\u00b3\3\2\2"+
		"\2\67\u00b6\3\2\2\29\u00b9\3\2\2\2;\u00bb\3\2\2\2=\u00bd\3\2\2\2?\u00c0"+
		"\3\2\2\2A\u00c3\3\2\2\2C\u00c6\3\2\2\2E\u00c9\3\2\2\2G\u00cb\3\2\2\2I"+
		"\u00cd\3\2\2\2K\u00cf\3\2\2\2M\u00d1\3\2\2\2O\u00d3\3\2\2\2Q\u00d5\3\2"+
		"\2\2S\u00d7\3\2\2\2UW\t\2\2\2VU\3\2\2\2WX\3\2\2\2XV\3\2\2\2XY\3\2\2\2"+
		"YZ\3\2\2\2Z[\b\2\2\2[\4\3\2\2\2\\]\7k\2\2]^\7h\2\2^\6\3\2\2\2_`\7g\2\2"+
		"`a\7n\2\2ab\7u\2\2bc\7g\2\2c\b\3\2\2\2de\7f\2\2ef\7q\2\2f\n\3\2\2\2gh"+
		"\7y\2\2hi\7j\2\2ij\7k\2\2jk\7n\2\2kl\7g\2\2l\f\3\2\2\2mn\7h\2\2no\7q\2"+
		"\2op\7t\2\2p\16\3\2\2\2qr\7t\2\2rs\7g\2\2st\7v\2\2tu\7w\2\2uv\7t\2\2v"+
		"w\7p\2\2w\20\3\2\2\2xy\7d\2\2yz\7t\2\2z{\7g\2\2{|\7c\2\2|}\7m\2\2}\22"+
		"\3\2\2\2~\177\7e\2\2\177\u0080\7q\2\2\u0080\u0081\7p\2\2\u0081\u0082\7"+
		"v\2\2\u0082\u0083\7k\2\2\u0083\u0084\7p\2\2\u0084\u0085\7w\2\2\u0085\u0086"+
		"\7g\2\2\u0086\24\3\2\2\2\u0087\u0088\7k\2\2\u0088\u0089\7p\2\2\u0089\u008a"+
		"\7v\2\2\u008a\26\3\2\2\2\u008b\u008d\t\3\2\2\u008c\u008b\3\2\2\2\u008d"+
		"\u008e\3\2\2\2\u008e\u008c\3\2\2\2\u008e\u008f\3\2\2\2\u008f\30\3\2\2"+
		"\2\u0090\u0094\t\4\2\2\u0091\u0093\t\5\2\2\u0092\u0091\3\2\2\2\u0093\u0096"+
		"\3\2\2\2\u0094\u0092\3\2\2\2\u0094\u0095\3\2\2\2\u0095\32\3\2\2\2\u0096"+
		"\u0094\3\2\2\2\u0097\u0098\7?\2\2\u0098\34\3\2\2\2\u0099\u009a\7-\2\2"+
		"\u009a\36\3\2\2\2\u009b\u009c\7/\2\2\u009c \3\2\2\2\u009d\u009e\7,\2\2"+
		"\u009e\"\3\2\2\2\u009f\u00a0\7\61\2\2\u00a0$\3\2\2\2\u00a1\u00a2\7\'\2"+
		"\2\u00a2&\3\2\2\2\u00a3\u00a4\7#\2\2\u00a4(\3\2\2\2\u00a5\u00a6\7\u0080"+
		"\2\2\u00a6*\3\2\2\2\u00a7\u00a8\7(\2\2\u00a8,\3\2\2\2\u00a9\u00aa\7`\2"+
		"\2\u00aa.\3\2\2\2\u00ab\u00ac\7~\2\2\u00ac\60\3\2\2\2\u00ad\u00ae\7>\2"+
		"\2\u00ae\u00af\7>\2\2\u00af\62\3\2\2\2\u00b0\u00b1\7@\2\2\u00b1\u00b2"+
		"\7@\2\2\u00b2\64\3\2\2\2\u00b3\u00b4\7?\2\2\u00b4\u00b5\7?\2\2\u00b5\66"+
		"\3\2\2\2\u00b6\u00b7\7#\2\2\u00b7\u00b8\7?\2\2\u00b88\3\2\2\2\u00b9\u00ba"+
		"\7>\2\2\u00ba:\3\2\2\2\u00bb\u00bc\7@\2\2\u00bc<\3\2\2\2\u00bd\u00be\7"+
		">\2\2\u00be\u00bf\7?\2\2\u00bf>\3\2\2\2\u00c0\u00c1\7@\2\2\u00c1\u00c2"+
		"\7?\2\2\u00c2@\3\2\2\2\u00c3\u00c4\7(\2\2\u00c4\u00c5\7(\2\2\u00c5B\3"+
		"\2\2\2\u00c6\u00c7\7~\2\2\u00c7\u00c8\7~\2\2\u00c8D\3\2\2\2\u00c9\u00ca"+
		"\7<\2\2\u00caF\3\2\2\2\u00cb\u00cc\7A\2\2\u00ccH\3\2\2\2\u00cd\u00ce\7"+
		"=\2\2\u00ceJ\3\2\2\2\u00cf\u00d0\7*\2\2\u00d0L\3\2\2\2\u00d1\u00d2\7+"+
		"\2\2\u00d2N\3\2\2\2\u00d3\u00d4\7}\2\2\u00d4P\3\2\2\2\u00d5\u00d6\7\177"+
		"\2\2\u00d6R\3\2\2\2\u00d7\u00d8\7.\2\2\u00d8T\3\2\2\2\6\2X\u008e\u0094"+
		"\3\b\2\2";
	public static final ATN _ATN =
		new ATNDeserializer().deserialize(_serializedATN.toCharArray());
	static {
		_decisionToDFA = new DFA[_ATN.getNumberOfDecisions()];
		for (int i = 0; i < _ATN.getNumberOfDecisions(); i++) {
			_decisionToDFA[i] = new DFA(_ATN.getDecisionState(i), i);
		}
	}
}