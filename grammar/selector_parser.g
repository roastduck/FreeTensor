parser grammar selector_parser;

options {
	tokenVocab = selector_lexer;
}

@parser::postinclude {
    #include <selector.h>
}

selector
	returns[Ref<Selector> s]:
	LeftBracket selector RightBracket {
        $s = $selector.s;
    }
	| Label {
        $s = Ref<LabelSelector>::make($Label.text);
    }
	| Id {
        $s = Ref<IDSelector>::make(ID::make(std::stoi($Id.text)));
    }
	| NodeTypeAssert {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::Assert);
    }
	| NodeTypeAssume {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::Assume);
    }
	| NodeTypeFor {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::For);
    }
	| NodeTypeIf {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::If);
    }
	| NodeTypeStmtSeq {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::StmtSeq);
    }
	| NodeTypeVarDef {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::VarDef);
    }
	| s1 = selector And s2 = selector {
        $s = Ref<BothSelector>::make($s1.s, $s2.s);
    }
	| s1 = selector Or s2 = selector {
        $s = Ref<EitherSelector>::make($s1.s, $s2.s);
    }
	| s1 = selector ChildArrow s2 = selector {
        $s = Ref<ChildSelector>::make($s1.s, $s2.s);
    }
	| s1 = selector DescendantArrow s2 = selector {
        $s = Ref<DescendantSelector>::make($s1.s, $s2.s);
    };
