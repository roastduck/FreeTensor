parser grammar selector_parser;

options {
	tokenVocab = selector_lexer;
}

@parser::postinclude {
    #include <selector.h>
}

leafSelector
	returns[Ref<LeafSelector> s]:
	| Label { std::vector<std::string> labels{$Label.text}; } (
		Label { labels.push_back($Label.text); }
	)* {
        $s = Ref<LabelSelector>::make(labels);
    }
	| Id {
        $s = Ref<IDSelector>::make(ID::make(std::stoi($Id.text)));
    }
	| TransformOp LeftBracket leafSelector { std::vector<Ref<LeafSelector>> srcs{$leafSelector.s}; }
		(
		Comma leafSelector { srcs.push_back($leafSelector.s); }
	)* RightBracket {
        $s = Ref<TransformedSelector>::make($TransformOp.text.substr(1), srcs);
    }
	| s1 = leafSelector And s2 = leafSelector {
        $s = Ref<BothLeafSelector>::make($s1.s, $s2.s);
    }
	| s1 = leafSelector Or s2 = leafSelector {
        $s = Ref<EitherLeafSelector>::make($s1.s, $s2.s);
    }
	| <assoc=right> callee = leafSelector CallerArrow caller = leafSelector {
        $s = Ref<BothLeafSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s));
    };

selector
	returns[Ref<Selector> s]:
	LeftBracket selector RightBracket {
        $s = $selector.s;
    }
	| leafSelector {
        $s = $leafSelector.s;
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
	| <assoc=right> child = selector ChildArrow parent = selector {
        $s = Ref<BothSelector>::make($child.s, Ref<ChildSelector>::make($parent.s));
    }
	| <assoc=right> descendant = selector DescendantArrow ancestor = selector {
        $s = Ref<BothSelector>::make($descendant.s, Ref<DescendantSelector>::make($ancestor.s));
    }
	| <assoc=right> callee = selector CallerArrow caller = leafSelector {
        $s = Ref<BothSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s));
    };
