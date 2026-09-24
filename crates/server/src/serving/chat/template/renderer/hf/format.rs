//! Chat-template content-shape detection and formatting options.
//!
//! Hugging Face templates expect message `content` either as a plain string or
//! as an OpenAI-style list of typed parts. Under
//! [`ChatTemplateContentFormatOption::Auto`],
//! `detect_chat_template_content_format` inspects the parsed template AST:
//!
//! - no loop over a message's `content` selects `String`;
//! - such a loop selects `OpenAi`;
//! - such a loop plus an `if` condition testing `content is string` selects
//!   `Preserve`, because the template handles both shapes.
//!
//! The analysis is name-based and ignores scoping: any assignment or loop
//! anywhere in the template (including macro bodies) counts, and only plain
//! variable targets are tracked.

use std::collections::{HashSet, VecDeque};
use std::fmt;
use std::str::FromStr;

use minijinja::machinery::ast::{Expr, ForLoop, Set, Stmt};
use minijinja::machinery::{WhitespaceConfig, parse};
use minijinja::syntax::SyntaxConfig;
use serde_with::{DeserializeFromStr, SerializeDisplay};

/// Resolved message-content shape passed to a compiled template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) enum ChatTemplateContentFormat {
    /// Content is a simple string.
    #[default]
    String,
    /// Content is a list of structured parts (OpenAI format).
    OpenAi,
    /// String messages remain strings and structured messages remain lists.
    Preserve,
}

/// Configurable chat-template content format selection.
///
/// `FromStr` accepts the `*_LITERAL` constants case-insensitively.
/// `ChatTemplateContentFormat::Preserve` has no option; only `Auto` detection
/// selects it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, DeserializeFromStr, SerializeDisplay)]
pub enum ChatTemplateContentFormatOption {
    /// Detect the format from the template source.
    #[default]
    Auto,
    /// Always flatten content into plain strings before rendering.
    String,
    /// Always pass content through in OpenAI-compatible structured form.
    OpenAi,
}

impl ChatTemplateContentFormatOption {
    /// Configuration literal for automatic content-shape detection.
    pub const AUTO_LITERAL: &str = "auto";
    /// Configuration literal for structured OpenAI content parts.
    pub const OPENAI_LITERAL: &str = "openai";
    /// Configuration literal for flattened string content.
    pub const STRING_LITERAL: &str = "string";
}

impl FromStr for ChatTemplateContentFormatOption {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.eq_ignore_ascii_case(Self::AUTO_LITERAL) {
            Ok(Self::Auto)
        } else if value.eq_ignore_ascii_case(Self::STRING_LITERAL) {
            Ok(Self::String)
        } else if value.eq_ignore_ascii_case(Self::OPENAI_LITERAL) {
            Ok(Self::OpenAi)
        } else {
            Err(format!(
                "invalid content format `{value}`; expected one of: {}, {}, {}",
                Self::AUTO_LITERAL,
                Self::STRING_LITERAL,
                Self::OPENAI_LITERAL
            ))
        }
    }
}

impl fmt::Display for ChatTemplateContentFormatOption {
    /// Writes the lowercase configuration literal.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => f.write_str(Self::AUTO_LITERAL),
            Self::String => f.write_str(Self::STRING_LITERAL),
            Self::OpenAi => f.write_str(Self::OPENAI_LITERAL),
        }
    }
}

/// Returns whether an expression references the named variable.
fn is_var_access(expr: &Expr, varname: &str) -> bool {
    matches!(expr, Expr::Var(v) if v.id == varname)
}

/// Returns whether an expression is the requested string constant.
fn is_const_str(expr: &Expr, value: &str) -> bool {
    matches!(expr, Expr::Const(c) if c.value.as_str() == Some(value))
}

/// Returns whether an expression accesses the requested variable attribute.
fn is_attr_access(expr: &Expr, varname: &str, key: &str) -> bool {
    match expr {
        Expr::GetItem(g) => is_var_access(&g.expr, varname) && is_const_str(&g.subscript_expr, key),
        Expr::GetAttr(g) => is_var_access(&g.expr, varname) && g.name == key,
        _ => false,
    }
}

/// Returns whether an expression references a variable or one of its elements.
///
/// With `key`, matches `varname.key` or `varname['key']`; without it, matches
/// `varname` itself. Filters, tests, and slices applied to the match are looked
/// through, so `messages|reverse` and `messages[1:]` still count as `messages`.
fn is_var_or_elems_access(expr: &Expr, varname: &str, key: Option<&str>) -> bool {
    match expr {
        Expr::Filter(f) => f
            .expr
            .as_ref()
            .is_some_and(|inner| is_var_or_elems_access(inner, varname, key)),
        Expr::Test(t) => is_var_or_elems_access(&t.expr, varname, key),
        Expr::Slice(s) => is_var_or_elems_access(&s.expr, varname, key),
        _ => key.map_or_else(
            || is_var_access(expr, varname),
            |key| is_attr_access(expr, varname, key),
        ),
    }
}

/// Traverses a template AST and collects assignments and loops in encounter order.
fn visit_stmt<'a>(
    stmt: &'a Stmt<'a>,
    assignments: &mut Vec<&'a Set<'a>>,
    loops: &mut Vec<&'a ForLoop<'a>>,
) {
    match stmt {
        Stmt::Template(t) => {
            for child in &t.children {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::ForLoop(fl) => {
            loops.push(fl);
            for child in &fl.body {
                visit_stmt(child, assignments, loops);
            }
            for child in &fl.else_body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::IfCond(ic) => {
            for child in &ic.true_body {
                visit_stmt(child, assignments, loops);
            }
            for child in &ic.false_body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::WithBlock(wb) => {
            for child in &wb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Set(set_stmt) => assignments.push(set_stmt),
        Stmt::SetBlock(sb) => {
            for child in &sb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::AutoEscape(ae) => {
            for child in &ae.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::FilterBlock(fb) => {
            for child in &fb.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Block(b) => {
            for child in &b.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::Macro(m) => {
            for child in &m.body {
                visit_stmt(child, assignments, loops);
            }
        }
        Stmt::CallBlock(cb) => {
            for child in &cb.macro_decl.body {
                visit_stmt(child, assignments, loops);
            }
        }
        _ => {}
    }
}

/// Collects every `set` statement and `for` loop in the template.
fn collect_assignments_and_loops<'a>(
    root: &'a Stmt<'a>,
) -> (Vec<&'a Set<'a>>, Vec<&'a ForLoop<'a>>) {
    let mut assignments = Vec::new();
    let mut loops = Vec::new();
    visit_stmt(root, &mut assignments, &mut loops);
    (assignments, loops)
}

/// Finds variables transitively assigned from a source variable or its elements.
///
/// Returns `varname` itself followed by every alias discovered breadth-first,
/// e.g. `loop_messages` for `{% set loop_messages = messages[1:] %}`.
fn iter_nodes_assign_var_or_elems(root: &Stmt<'_>, varname: &str) -> Vec<String> {
    let (assignments, _) = collect_assignments_and_loops(root);

    let mut discovered = vec![varname.to_string()];
    let mut seen = HashSet::from([varname.to_string()]);
    let mut related = VecDeque::from([varname.to_string()]);

    while let Some(related_varname) = related.pop_front() {
        for assign in &assignments {
            let Expr::Var(lhs) = &assign.target else {
                continue;
            };

            if is_var_or_elems_access(&assign.expr, &related_varname, None) {
                let lhs_name = lhs.id.to_string();
                if seen.insert(lhs_name.clone()) {
                    discovered.push(lhs_name.clone());
                    if lhs_name != related_varname {
                        related.push_back(lhs_name);
                    }
                }
            }
        }
    }

    discovered
}

/// Finds loop variables that iterate over the template's message collection.
fn iter_nodes_assign_messages_item(root: &Stmt<'_>) -> Vec<String> {
    let message_varnames = iter_nodes_assign_var_or_elems(root, "messages");
    let (_, loops) = collect_assignments_and_loops(root);

    let mut discovered = Vec::new();
    let mut seen = HashSet::new();

    for loop_ast in loops {
        let Expr::Var(target) = &loop_ast.target else {
            continue;
        };

        if message_varnames
            .iter()
            .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, None))
        {
            let target_name = target.id.to_string();
            if seen.insert(target_name.clone()) {
                discovered.push(target_name);
            }
        }
    }

    discovered
}

/// Returns whether some loop iterates over a message item's `content`.
fn has_content_item_loop(root: &Stmt<'_>) -> bool {
    let message_varnames = iter_nodes_assign_messages_item(root);
    let (_, loops) = collect_assignments_and_loops(root);

    loops.into_iter().any(|loop_ast| {
        matches!(loop_ast.target, Expr::Var(_))
            && message_varnames
                .iter()
                .any(|varname| is_var_or_elems_access(&loop_ast.iter, varname, Some("content")))
    })
}

/// Returns whether an expression tests message content for string representation.
fn expression_tests_content_string(expr: &Expr<'_>, message_varnames: &[String]) -> bool {
    match expr {
        Expr::Test(test) => {
            (test.name == "string"
                && message_varnames
                    .iter()
                    .any(|varname| is_var_or_elems_access(&test.expr, varname, Some("content"))))
                || expression_tests_content_string(&test.expr, message_varnames)
        }
        Expr::UnaryOp(unary) => expression_tests_content_string(&unary.expr, message_varnames),
        Expr::BinOp(binary) => {
            expression_tests_content_string(&binary.left, message_varnames)
                || expression_tests_content_string(&binary.right, message_varnames)
        }
        Expr::Compare(compare) => {
            expression_tests_content_string(&compare.expr, message_varnames)
                || compare
                    .ops
                    .iter()
                    .any(|op| expression_tests_content_string(&op.expr, message_varnames))
        }
        Expr::IfExpr(if_expr) => {
            expression_tests_content_string(&if_expr.test_expr, message_varnames)
                || expression_tests_content_string(&if_expr.true_expr, message_varnames)
                || if_expr
                    .false_expr
                    .as_ref()
                    .is_some_and(|expr| expression_tests_content_string(expr, message_varnames))
        }
        _ => false,
    }
}

/// Returns whether a statement subtree tests message content for string representation.
///
/// Only `if` statement conditions are inspected; expressions in `set`
/// statements, output blocks, and loop filters are not.
fn statement_tests_content_string(stmt: &Stmt<'_>, message_varnames: &[String]) -> bool {
    let children_test = |children: &[Stmt<'_>]| {
        children
            .iter()
            .any(|child| statement_tests_content_string(child, message_varnames))
    };
    match stmt {
        Stmt::Template(template) => children_test(&template.children),
        Stmt::ForLoop(for_loop) => {
            children_test(&for_loop.body) || children_test(&for_loop.else_body)
        }
        Stmt::IfCond(if_cond) => {
            expression_tests_content_string(&if_cond.expr, message_varnames)
                || children_test(&if_cond.true_body)
                || children_test(&if_cond.false_body)
        }
        Stmt::WithBlock(with_block) => children_test(&with_block.body),
        Stmt::SetBlock(set_block) => children_test(&set_block.body),
        Stmt::AutoEscape(auto_escape) => children_test(&auto_escape.body),
        Stmt::FilterBlock(filter_block) => children_test(&filter_block.body),
        Stmt::Block(block) => children_test(&block.body),
        Stmt::Macro(macro_stmt) => children_test(&macro_stmt.body),
        Stmt::CallBlock(call_block) => children_test(&call_block.macro_decl.body),
        _ => false,
    }
}

/// Returns whether some `if` condition tests a message item's `content` with
/// `is string`.
fn has_content_string_test(root: &Stmt<'_>) -> bool {
    let message_varnames = iter_nodes_assign_messages_item(root);
    statement_tests_content_string(root, &message_varnames)
}

/// Detects the content format expected by a Jinja2 chat template from AST
/// analysis.
pub(super) fn detect_chat_template_content_format(template: &str) -> ChatTemplateContentFormat {
    let ast = match parse(
        template,
        "template",
        SyntaxConfig {},
        WhitespaceConfig::default(),
    ) {
        Ok(ast) => ast,
        // `CompiledChatTemplate::new` compiles the same source right after
        // detection, so a template that fails to parse here fails renderer
        // construction there.
        Err(_) => return ChatTemplateContentFormat::String,
    };

    if has_content_item_loop(&ast) {
        if has_content_string_test(&ast) {
            ChatTemplateContentFormat::Preserve
        } else {
            ChatTemplateContentFormat::OpenAi
        }
    } else {
        ChatTemplateContentFormat::String
    }
}
