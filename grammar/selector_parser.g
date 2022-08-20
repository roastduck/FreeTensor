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
	| <assoc=right> l1 = leafSelector CallerArrow l2 = leafSelector {
        $s = Ref<CallerSelector>::make($l1.s, $l2.s);
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
	| child = selector ChildArrow parent = selector {
        $s = Ref<ChildSelector>::make($parent.s, $child.s);
    }
	| descendant = selector DescendantArrow ancestor = selector {
        $s = Ref<DescendantSelector>::make($ancestor.s, $descendant.s);
    };
