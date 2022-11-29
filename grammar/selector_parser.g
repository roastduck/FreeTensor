parser grammar selector_parser;

options {
	tokenVocab = selector_lexer;
}

@parser::postinclude {
    #include <selector.h>
}

leafSelector returns[Ref<LeafSelector> s]
	: LeftParen leafSelector RightParen {
        $s = $leafSelector.s;
    }
	| Label {
        $s = Ref<LabelSelector>::make($Label.text);
    }
	| Id {
        $s = Ref<IDSelector>::make(ID::make(std::stoi($Id.text.substr(1))));
    }
	| TransformOp LeftBracket leafSelector { std::vector<Ref<LeafSelector>> srcs{$leafSelector.s}; }
		(
		Comma leafSelector { srcs.push_back($leafSelector.s); }
	)* RightBracket {
        $s = Ref<TransformedSelector>::make($TransformOp.text.substr(1), srcs);
    }
    | RootCall {
        $s = Ref<RootCallSelector>::make();
    }
    | Not sub=leafSelector {
        $s = Ref<NotLeafSelector>::make($sub.s);
    }
	| s1 = leafSelector And s2 = leafSelector {
        $s = Ref<BothLeafSelector>::make($s1.s, $s2.s);
    }
	| s1 = leafSelector Or s2 = leafSelector {
        $s = Ref<EitherLeafSelector>::make($s1.s, $s2.s);
    }
	| DirectCallerArrow caller = leafSelector {
        $s = Ref<DirectCallerSelector>::make($caller.s);
    }
	| <assoc=right> callee = leafSelector DirectCallerArrow caller = leafSelector {
        $s = Ref<BothLeafSelector>::make($callee.s, Ref<DirectCallerSelector>::make($caller.s));
    }
	| CallerArrow caller = leafSelector {
        $s = Ref<CallerSelector>::make($caller.s);
    }
	| <assoc=right> callee = leafSelector CallerArrow caller = leafSelector {
        $s = Ref<BothLeafSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s));
    }
	| DirectCallerArrow LeftParen middle = leafSelector DirectCallerArrow RightParen Star caller = leafSelector {
        $s = Ref<CallerSelector>::make($caller.s, $middle.s);
    }
	| <assoc=right> callee = leafSelector DirectCallerArrow LeftParen middle = leafSelector DirectCallerArrow RightParen Star caller = leafSelector {
        $s = Ref<BothLeafSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s, $middle.s));
    };

selector returns[Ref<Selector> s]
	: LeftParen selector RightParen {
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
	| NodeTypeStore {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::Store);
    }
	| NodeTypeReduceTo {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::ReduceTo);
    }
	| NodeTypeEval {
        $s = Ref<NodeTypeSelector>::make(ASTNodeType::Eval);
    }
    | RootNode {
        $s = Ref<RootNodeSelector>::make();
    }
    | Not sub=selector {
        $s = Ref<NotSelector>::make($sub.s);
    }
	| s1 = selector And s2 = selector {
        $s = Ref<BothSelector>::make($s1.s, $s2.s);
    }
	| s1 = selector Or s2 = selector {
        $s = Ref<EitherSelector>::make($s1.s, $s2.s);
    }
	| ChildArrow parent = selector {
        $s = Ref<ChildSelector>::make($parent.s);
    }
    | ParentArrow chlid = selector {
        $s = Ref<ParentSelector>::make($child.s);
    }
	| <assoc=right> child = selector ChildArrow parent = selector {
        $s = Ref<BothSelector>::make($child.s, Ref<ChildSelector>::make($parent.s));
    }
    | <assoc=right> parent = selector ParentArrow child = selector {
        $s = Ref<BothSelector>::make($parent.s, Ref<ParentSelector>::make($child.s));
    }
	| DescendantArrow ancestor = selector {
        $s = Ref<DescendantSelector>::make($ancestor.s);
    }
    | AncestorArrow descendant = selector {
        $s = Ref<AncestorSelector>::make($descendant.s);
    }
	| <assoc=right> descendant = selector DescendantArrow ancestor = selector {
        $s = Ref<BothSelector>::make($descendant.s, Ref<DescendantSelector>::make($ancestor.s));
    }
    | <assoc=right> ancestor = selector AncestorArrow descendant = selector {
        $s = Ref<BothSelector>::make($ancestor.s, Ref<AncestorSelector>::make($descendant.s));
    }
    | ChildArrow LeftParen middle = selector ChildArrow RightParen Star ancestor = selector {
        $s = Ref<DescendantSelector>::make($ancestor.s, $middle.s);
    }
    | ParentArrow LeftParan middle = selector ParentArrow RightParen Star descendant = selector {
        $s = Ref<AncestorSelector>::make($descendant.s, $middle.s);
    }
    | <assoc=right> descendant = selector ChildArrow LeftParen middle = selector ChildArrow RightParen Star ancestor = selector {
        $s = Ref<BothSelector>::make($descendant.s, Ref<DescendantSelector>::make($ancestor.s, $middle.s));
    }
    | <assoc=right> ancestor = selector ParentArrow LeftParen middle = selector ParentArrow RightParen Star descendant = selector {
        $s = Ref<BothSelector>::make($ancestor.s, Ref<AncestorSelector>::make($descendant.s, $middle.s));
    }
	| <assoc=right> callee = selector DirectCallerArrow caller = leafSelector {
        $s = Ref<BothSelector>::make($callee.s, Ref<DirectCallerSelector>::make($caller.s));
    }
	| <assoc=right> callee = selector CallerArrow caller = leafSelector {
        $s = Ref<BothSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s));
    }
	| <assoc=right> callee = selector DirectCallerArrow LeftParen mid = leafSelector DirectCallerArrow RightParen Star caller = leafSelector {
        $s = Ref<BothSelector>::make($callee.s, Ref<CallerSelector>::make($caller.s, $mid.s));
    };
