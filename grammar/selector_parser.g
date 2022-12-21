parser grammar selector_parser;

options {
	tokenVocab = selector_lexer;
}

@parser::postinclude {
    #include <selector.h>
}

all returns[Ref<Selector> s]
    : selector EOF {
        $s = $selector.s;
    };

metadataSelector returns[Ref<MetadataSelector> s]
    : metadataSelectorFactor {
        $s = $metadataSelectorFactor.s;
    }
	| s1=metadataSelector And s2=metadataSelector {
        $s = Ref<BothMetadataSelector>::make($s1.s, $s2.s);
    }
	| s1=metadataSelector Or s2=metadataSelector {
        $s = Ref<EitherMetadataSelector>::make($s1.s, $s2.s);
    };

metadataSelectorFactor returns[Ref<MetadataSelector> s]
	: LeftParen metadataSelector RightParen {
        $s = $metadataSelector.s;
    }
    | Not sub=metadataSelectorFactor {
        $s = Ref<NotMetadataSelector>::make($sub.s);
    }
	| Label {
        $s = Ref<LabelSelector>::make($Label.text);
    }
	| Id {
        $s = Ref<IDSelector>::make(ID::make(std::stoi($Id.text.substr(1))));
    }
	| TransformOp LeftBracket metadataSelector { std::vector<Ref<MetadataSelector>> srcs{$metadataSelector.s}; }
		(
		Comma metadataSelector { srcs.push_back($metadataSelector.s); }
	)* RightBracket {
        $s = Ref<TransformedSelector>::make($TransformOp.text.substr(1), srcs);
    }
    | RootCall {
        $s = Ref<RootCallSelector>::make();
    }
    | rhs=metadataSelectorImplicitAnd {
        $s = $rhs.s;
    }
    | <assoc=right> lhs=metadataSelectorFactor rhs=metadataSelectorImplicitAnd {
        $s = Ref<BothMetadataSelector>::make($lhs.s, $rhs.s);
    };

// `A<~B` can be viewed as `A&<~B`. This node represents the `<~B` part.
metadataSelectorImplicitAnd returns[Ref<MetadataSelector> s]
	: DirectCallerArrow caller=metadataSelectorFactor {
        $s = Ref<DirectCallerSelector>::make($caller.s);
    }
	| CallerArrow caller=metadataSelectorFactor {
        $s = Ref<CallerSelector>::make($caller.s);
    }
	| DirectCallerArrow LeftParen middle=metadataSelectorFactor DirectCallerArrow RightParen Star caller=metadataSelectorFactor {
        $s = Ref<CallerSelector>::make($caller.s, $middle.s);
    };

selector returns[Ref<Selector> s]
    : selectorFactor {
        $s = $selectorFactor.s;
    }
	| s1=selector And s2=selector {
        $s = Ref<BothSelector>::make($s1.s, $s2.s);
    }
	| s1=selector Or s2=selector {
        $s = Ref<EitherSelector>::make($s1.s, $s2.s);
    };

selectorFactor returns[Ref<Selector> s]
	: LeftParen selector RightParen {
        $s = $selector.s;
    }
    | Not sub=selectorFactor {
        $s = Ref<NotSelector>::make($sub.s);
    }
	| metadataSelector {
        $s = $metadataSelector.s;
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
    | LeafNode {
        $s = Ref<LeafNodeSelector>::make();
    }
    | rhs=selectorImplicitAnd {
        $s = $rhs.s;
    }
    | <assoc=right> lhs=selectorFactor rhs=selectorImplicitAnd {
        $s = Ref<BothSelector>::make($lhs.s, $rhs.s);
    };

// `A<-B` can be viewed as `A&<-B`. This node represents the `<-B` part.
selectorImplicitAnd returns[Ref<Selector> s]
    : metadataSelectorImplicitAnd {
        $s = $metadataSelectorImplicitAnd.s;
    }
	| ChildArrow parent=selectorFactor {
        $s = Ref<ChildSelector>::make($parent.s);
    }
    | ParentArrow child=selectorFactor {
        $s = Ref<ParentSelector>::make($child.s);
    }
	| DescendantArrow ancestor=selectorFactor {
        $s = Ref<DescendantSelector>::make($ancestor.s);
    }
    | AncestorArrow descendant=selectorFactor {
        $s = Ref<AncestorSelector>::make($descendant.s);
    }
    | ChildArrow LeftParen middle=selectorFactor ChildArrow RightParen Star ancestor=selectorFactor {
        $s = Ref<DescendantSelector>::make($ancestor.s, $middle.s);
    }
    | ParentArrow LeftParen middle=selectorFactor ParentArrow RightParen Star descendant=selectorFactor {
        $s = Ref<AncestorSelector>::make($descendant.s, $middle.s);
    };
