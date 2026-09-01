//! Parse model-emitted tool calls into OpenAI `tool_calls`.
//!
//! DSV4 jinja asks for DSML (`<｜DSML｜tool_calls>` / `invoke` / `parameter`).
//! GLM-5.3 / GLM-4.7 jinja asks for `<tool_call>name<arg_key>…<arg_value>…`.
//! Local agents also leak Shepherd/Hermes/Qwen XML into `content`. We convert
//! all of those so the client sees structured `tool_calls` instead of raw tags.

use serde_json::{Map, Value};

const DSML: &str = "｜DSML｜";

const START_MARKERS: &[&str] = &[
    "<｜DSML｜tool_calls>",
    "<｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "<tool_call>",
    "<tools_call>",
    "<function=",
    "<|tool▁calls▁begin|>",
    "[tool_call",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedTools {
    pub content: String,
    pub calls: Vec<ToolCall>,
}

/// Incremental splitter: stream prose, hold a tool-call block until `finish`.
pub struct ToolCallParser {
    hold: String,
    in_block: bool,
    seq: u32,
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self {
            hold: String::new(),
            in_block: false,
            seq: 0,
        }
    }

    pub fn push(&mut self, piece: &str) -> Option<String> {
        if piece.is_empty() {
            return None;
        }
        self.hold.push_str(piece);
        if self.in_block {
            return None;
        }
        if let Some(idx) = find_start(&self.hold) {
            let before = self.hold[..idx].to_string();
            self.hold.drain(..idx);
            self.in_block = true;
            return nonempty(before);
        }
        let keep = longest_start_prefix(&self.hold);
        let emit_len = self.hold.len() - keep;
        if emit_len == 0 {
            return None;
        }
        let emit = self.hold[..emit_len].to_string();
        self.hold.drain(..emit_len);
        nonempty(emit)
    }

    pub fn finish(&mut self) -> ParsedTools {
        let rest = std::mem::take(&mut self.hold);
        self.in_block = false;
        let mut parsed = parse_tool_calls(&rest);
        for call in &mut parsed.calls {
            if call.id.is_empty() {
                self.seq += 1;
                call.id = format!("call_{}", self.seq);
            }
        }
        parsed
    }
}

pub fn parse_tool_calls(text: &str) -> ParsedTools {
    if text.is_empty() {
        return ParsedTools::default();
    }

    let mut content = String::new();
    let mut calls = Vec::new();
    let mut idx = 0;
    let mut seq = 0u32;

    while idx < text.len() {
        let rel = match find_start(&text[idx..]) {
            Some(r) => r,
            None => {
                content.push_str(&text[idx..]);
                break;
            }
        };
        let start = idx + rel;
        content.push_str(&text[idx..start]);

        let (end, block) = extract_block(&text[start..]);
        idx = start + end;

        let mut found = parse_block(block);
        if found.is_empty() {
            content.push_str(block);
            continue;
        }
        for call in &mut found {
            if call.id.is_empty() {
                seq += 1;
                call.id = format!("call_{seq}");
            }
        }
        calls.extend(found);
    }

    ParsedTools {
        content: trim_tool_gaps(&content),
        calls,
    }
}

fn parse_block(block: &str) -> Vec<ToolCall> {
    let dsml = parse_dsml(block);
    if !dsml.is_empty() {
        return dsml;
    }
    if let Some(tc) = parse_function_eq(block) {
        return vec![tc];
    }
    if let Some(tc) = parse_tool_name_xml(block) {
        return vec![tc];
    }
    if let Some(tc) = parse_json_blob(strip_known_wrappers(block)) {
        return vec![tc];
    }
    if let Some(tc) = parse_glm_arg_key(block) {
        return vec![tc];
    }
    Vec::new()
}

fn parse_dsml(block: &str) -> Vec<ToolCall> {
    let invoke_open = format!("<{DSML}invoke");
    let invoke_close = format!("</{DSML}invoke>");
    let param_open = format!("<{DSML}parameter");
    let param_close = format!("</{DSML}parameter>");

    let mut out = Vec::new();
    let mut search = 0;
    while let Some(rel) = block[search..].find(&invoke_open) {
        let start = search + rel;
        let after_open = start + invoke_open.len();
        let Some(gt) = block[after_open..].find('>') else {
            break;
        };
        let header = &block[after_open..after_open + gt];
        let Some(name) = attr_value(header, "name") else {
            search = after_open;
            continue;
        };
        let body_start = after_open + gt + 1;
        let body_end = block[body_start..]
            .find(&invoke_close)
            .map(|i| body_start + i)
            .unwrap_or(block.len());
        let body = &block[body_start..body_end];

        let mut map = Map::new();
        let mut p = 0;
        while let Some(pr) = body[p..].find(&param_open) {
            let ps = p + pr;
            let after = ps + param_open.len();
            let Some(pgt) = body[after..].find('>') else {
                break;
            };
            let pheader = &body[after..after + pgt];
            let Some(pname) = attr_value(pheader, "name") else {
                p = after;
                continue;
            };
            let is_string = attr_value(pheader, "string")
                .map(|s| s.eq_ignore_ascii_case("true"))
                .unwrap_or(false);
            let val_start = after + pgt + 1;
            let val_end = body[val_start..]
                .find(&param_close)
                .map(|i| val_start + i)
                .unwrap_or(body.len());
            let raw = body[val_start..val_end].to_string();
            map.insert(pname, param_value(&raw, is_string));
            p = val_end + param_close.len().min(body.len().saturating_sub(val_end));
        }

        out.push(ToolCall {
            id: String::new(),
            name,
            arguments: Value::Object(map).to_string(),
        });
        search = body_end + invoke_close.len().min(block.len().saturating_sub(body_end));
    }
    out
}

fn parse_function_eq(block: &str) -> Option<ToolCall> {
    let open = "<function=";
    let start = block.find(open)?;
    let name_start = start + open.len();
    let gt = block[name_start..].find('>')?;
    let name = block[name_start..name_start + gt].trim().to_string();
    if name.is_empty() {
        return None;
    }
    let body_start = name_start + gt + 1;
    let body_end = block[body_start..]
        .find("</function>")
        .map(|i| body_start + i)
        .unwrap_or(block.len());
    let body = block[body_start..body_end].trim();
    let arguments = function_body_to_args(body);
    Some(ToolCall {
        id: String::new(),
        name,
        arguments,
    })
}

fn parse_tool_name_xml(block: &str) -> Option<ToolCall> {
    let mut name = None;
    let mut search = 0;
    let open = "<tool_name>";
    let close = "</tool_name>";
    while let Some(rel) = block[search..].find(open) {
        let s = search + rel + open.len();
        let Some(e) = block[s..].find(close) else {
            break;
        };
        let candidate = block[s..s + e].trim();
        if !candidate.is_empty() {
            name = Some(candidate.to_string());
            break;
        }
        search = s + e + close.len();
    }
    let name = name?;

    let arguments = if let Some(p) = extract_tag(block, "parameters") {
        let p = p.trim();
        if p.is_empty() {
            "{}".into()
        } else if looks_like_json(p) {
            normalize_args_json(p)
        } else if let Some(from_children) = parameters_from_children(p) {
            from_children
        } else {
            normalize_args_json(p)
        }
    } else if let Some(from_children) = parameters_from_children(block) {
        from_children
    } else {
        "{}".into()
    };

    Some(ToolCall {
        id: String::new(),
        name,
        arguments,
    })
}

fn parameters_from_children(body: &str) -> Option<String> {
    let mut map = Map::new();
    let mut i = 0;
    while i < body.len() {
        let rest = &body[i..];
        let named = rest.find("<parameter name=\"");
        let qwen = rest.find("<parameter=");
        let (kind, rel) = match (named, qwen) {
            (Some(a), Some(b)) if a <= b => ("name", a),
            (Some(a), None) => ("name", a),
            (_, Some(b)) => ("eq", b),
            _ => break,
        };
        let start = i + rel;
        let (key, body_start) = if kind == "name" {
            let after = start + "<parameter name=\"".len();
            let q = body[after..].find('"')?;
            let key = body[after..after + q].to_string();
            let gt = body[after + q..].find('>')?;
            (key, after + q + gt + 1)
        } else {
            let after = start + "<parameter=".len();
            let gt = body[after..].find('>')?;
            let key = body[after..after + gt].trim().to_string();
            (key, after + gt + 1)
        };
        if key.is_empty() {
            i = body_start;
            continue;
        }
        let close = "</parameter>";
        let end = body[body_start..]
            .find(close)
            .map(|x| body_start + x)
            .unwrap_or(body.len());
        let raw = body[body_start..end].trim();
        map.insert(key, param_value(raw, !looks_like_json(raw)));
        i = end + close.len().min(body.len().saturating_sub(end));
    }
    if map.is_empty() {
        None
    } else {
        Some(Value::Object(map).to_string())
    }
}

/// GLM-4.6/4.7/5.3 native tool markup.
/// Compact: `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>`
/// Newline: `<tool_call>name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>`
///
/// Name must look like a tool identifier so prose that happens to contain a
/// `<tool_call>` tag (shepherd leak parser) is not turned into a call.
fn parse_glm_arg_key(block: &str) -> Option<ToolCall> {
    let inner = strip_known_wrappers(block).trim();
    if inner.is_empty() {
        return None;
    }
    // Qwen `<function=` / `<tool_name>` / Hermes JSON: let those parsers win.
    if inner.starts_with('<') || inner.starts_with('{') || inner.starts_with('[') {
        return None;
    }

    const KEY_OPEN: &str = "<arg_key>";
    const KEY_CLOSE: &str = "</arg_key>";
    const VAL_OPEN: &str = "<arg_value>";
    const VAL_CLOSE: &str = "</arg_value>";

    let (name_raw, rest) = match inner.find(KEY_OPEN) {
        Some(idx) => (&inner[..idx], &inner[idx..]),
        None => {
            if inner.contains(KEY_CLOSE) || inner.contains(VAL_OPEN) || inner.contains(VAL_CLOSE) {
                return None;
            }
            (inner, "")
        }
    };
    let name = name_raw.trim();
    if !is_glm_tool_name(name) {
        return None;
    }

    let mut map = Map::new();
    let mut p = 0;
    while let Some(rel) = rest[p..].find(KEY_OPEN) {
        let ks = p + rel + KEY_OPEN.len();
        let Some(ke) = rest[ks..].find(KEY_CLOSE) else {
            break;
        };
        let key = rest[ks..ks + ke].trim();
        let after_key = ks + ke + KEY_CLOSE.len();
        let Some(vs_rel) = rest[after_key..].find(VAL_OPEN) else {
            break;
        };
        let vs = after_key + vs_rel + VAL_OPEN.len();
        let Some(ve) = rest[vs..].find(VAL_CLOSE) else {
            break;
        };
        if !key.is_empty() {
            let raw = &rest[vs..vs + ve];
            map.insert(key.to_string(), param_value(raw, false));
        }
        p = vs + ve + VAL_CLOSE.len();
    }

    Some(ToolCall {
        id: String::new(),
        name: name.to_string(),
        arguments: Value::Object(map).to_string(),
    })
}

fn is_glm_tool_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if name.len() > 128 {
        return false;
    }
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
}

fn parse_json_blob(s: &str) -> Option<ToolCall> {
    let s = strip_code_fence(s.trim());
    if s.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(s).ok()?;
    let obj = v.as_object()?;

    if let Some(func) = obj.get("function").and_then(|x| x.as_object()) {
        let name = func.get("name")?.as_str()?.trim().to_string();
        if name.is_empty() {
            return None;
        }
        let arguments = args_from_value(func.get("arguments"));
        let id = obj
            .get("id")
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        return Some(ToolCall {
            id,
            name,
            arguments,
        });
    }

    let name = obj.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let arguments = args_from_value(obj.get("arguments"));
    Some(ToolCall {
        id: String::new(),
        name,
        arguments,
    })
}

fn args_from_value(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => "{}".into(),
        Some(Value::String(s)) => {
            let s = s.trim();
            if s.is_empty() {
                "{}".into()
            } else if looks_like_json(s) {
                normalize_args_json(s)
            } else {
                s.to_string()
            }
        }
        Some(other) => other.to_string(),
    }
}

fn function_body_to_args(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return "{}".into();
    }
    if looks_like_json(body) {
        return normalize_args_json(body);
    }
    if let Some(from_children) = parameters_from_children(body) {
        return from_children;
    }
    normalize_args_json(body)
}

fn normalize_args_json(s: &str) -> String {
    match serde_json::from_str::<Value>(s) {
        Ok(Value::Object(_)) | Ok(Value::Array(_)) => {
            serde_json::from_str::<Value>(s).unwrap().to_string()
        }
        Ok(other) => other.to_string(),
        Err(_) => {
            if looks_like_json(s) {
                s.trim().to_string()
            } else {
                "{}".into()
            }
        }
    }
}

fn param_value(raw: &str, is_string: bool) -> Value {
    let raw = raw.trim();
    if is_string {
        return Value::String(raw.to_string());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn attr_value(header: &str, key: &str) -> Option<String> {
    let pat = format!("{key}=\"");
    let i = header.find(&pat)?;
    let start = i + pat.len();
    let end = header[start..].find('"')?;
    Some(header[start..start + end].to_string())
}

fn extract_tag<'a>(text: &'a str, tag: &str) -> Option<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let s = text.find(&open)? + open.len();
    let e = text[s..].find(&close)?;
    Some(&text[s..s + e])
}

fn extract_block(from_start: &str) -> (usize, &str) {
    let closer = if from_start.starts_with("<｜DSML｜tool_calls>") {
        Some("</｜DSML｜tool_calls>")
    } else if from_start.starts_with("<｜DSML｜function_calls>") {
        Some("</｜DSML｜function_calls>")
    } else if from_start.starts_with("<｜DSML｜invoke") {
        Some("</｜DSML｜invoke>")
    } else if from_start.starts_with("<tool_call>") {
        Some("</tool_call>")
    } else if from_start.starts_with("<tools_call>") {
        Some("</tools_call>")
    } else if from_start.starts_with("<function=") {
        Some("</function>")
    } else if from_start.starts_with("<|tool▁calls▁begin|>") {
        Some("<|tool▁calls▁end|>")
    } else if from_start.starts_with("[tool_call") {
        Some("[/tool_call]")
    } else {
        None
    };

    if let Some(close) = closer {
        if let Some(rel) = from_start.find(close) {
            let end = rel + close.len();
            return (end, &from_start[..end]);
        }
    }
    (from_start.len(), from_start)
}

fn find_start(text: &str) -> Option<usize> {
    START_MARKERS.iter().filter_map(|m| text.find(m)).min()
}

fn longest_start_prefix(hay: &str) -> usize {
    let mut keep = 0;
    for marker in START_MARKERS {
        let max = hay.len().min(marker.len().saturating_sub(1));
        for n in (1..=max).rev() {
            if hay.as_bytes().ends_with(&marker.as_bytes()[..n]) {
                keep = keep.max(n);
                break;
            }
        }
    }
    keep
}

fn strip_known_wrappers(block: &str) -> &str {
    let mut s = block.trim();
    for (open, close) in [
        ("<tool_call>", "</tool_call>"),
        ("<tools_call>", "</tools_call>"),
        ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"),
        ("<｜DSML｜function_calls>", "</｜DSML｜function_calls>"),
        ("<|tool▁calls▁begin|>", "<|tool▁calls▁end|>"),
    ] {
        if let Some(rest) = s.strip_prefix(open) {
            s = rest;
        }
        if let Some(rest) = s.strip_suffix(close) {
            s = rest;
        }
        s = s.trim();
    }
    s
}

fn strip_code_fence(s: &str) -> &str {
    let s = s.trim();
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s);
    s.strip_suffix("```").unwrap_or(s).trim()
}

fn looks_like_json(s: &str) -> bool {
    let s = s.trim();
    (s.starts_with('{') && s.ends_with('}')) || (s.starts_with('[') && s.ends_with(']'))
}

fn trim_tool_gaps(s: &str) -> String {
    let t = s.trim();
    if t.is_empty() {
        return String::new();
    }
    t.split('\n')
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn nonempty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_dsml_get_history() {
        let text = concat!(
            "ok\n\n",
            "<｜DSML｜tool_calls>\n",
            "<｜DSML｜invoke name=\"get_history\">\n",
            "<｜DSML｜parameter name=\"project_name\" string=\"true\">test</｜DSML｜parameter>\n",
            "<｜DSML｜parameter name=\"limit\" string=\"false\">10</｜DSML｜parameter>\n",
            "</｜DSML｜invoke>\n",
            "</｜DSML｜tool_calls>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.content, "ok");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "test");
        assert_eq!(v["limit"], 10);
    }

    #[test]
    fn parse_task323_tool_name_xml() {
        let text = concat!(
            "<tool_call>\n",
            "<tool_name>get_history</tool_name>\n",
            "<tool_name/>\n",
            "<parameters>\n",
            "</parameters>\n",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_history");
        assert_eq!(p.calls[0].arguments, "{}");
        assert!(p.content.is_empty());
    }

    #[test]
    fn parse_tool_name_with_json_params() {
        let text = concat!(
            "<tool_call>\n",
            "<tool_name>get_history</tool_name>\n",
            "<parameters>\n",
            "{\"project_name\":\"test\"}\n",
            "</parameters>\n",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "test");
    }

    #[test]
    fn parse_hermes_json() {
        let text = r#"<tool_call>
{"name":"get_history","arguments":{"project_name":"test"}}
</tool_call>"#;
        let p = parse_tool_calls(text);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "test");
    }

    #[test]
    fn parse_function_eq() {
        let text = r#"<function=bash>{"command":"ls"}</function>"#;
        let p = parse_tool_calls(text);
        assert_eq!(p.calls[0].name, "bash");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["command"], "ls");
    }

    #[test]
    fn parse_qwen38_tool_call_block() {
        let text = r#"<tool_call>
<function=get_history>
<parameter=project_name>
golbang
</parameter>
<parameter=limit>
5
</parameter>
</function>
</tool_call>"#;
        let p = parse_tool_calls(text);
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "golbang");
        assert_eq!(v["limit"], "5");
    }

    #[test]
    fn incremental_holds_partial_then_finishes() {
        let mut p = ToolCallParser::new();
        assert_eq!(p.push("hello ").as_deref(), Some("hello "));
        assert!(p.push("<tool").is_none());
        assert!(p
            .push("_call>\n<tool_name>get_history</tool_name>\n")
            .is_none());
        assert!(p
            .push("<parameters>{\"project_name\":\"test\"}</parameters>\n")
            .is_none());
        assert!(p.push("</tool_call>").is_none());
        let done = p.finish();
        assert!(done.content.is_empty());
        assert_eq!(done.calls.len(), 1);
        assert_eq!(done.calls[0].name, "get_history");
        assert_eq!(done.calls[0].id, "call_1");
    }

    #[test]
    fn prose_without_tools_passes_through() {
        let mut p = ToolCallParser::new();
        assert_eq!(p.push("hi").as_deref(), Some("hi"));
        let done = p.finish();
        assert!(done.calls.is_empty());
        assert!(done.content.is_empty());
    }

    #[test]
    fn parse_glm53_compact_arg_key() {
        let text = concat!(
            "<tool_call>get_history",
            "<arg_key>project_name</arg_key><arg_value>golbang</arg_value>",
            "<arg_key>limit</arg_key><arg_value>5</arg_value>",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.calls.len(), 1, "content leaked: {:?}", p.content);
        assert!(
            p.content.is_empty(),
            "glm markup must not leak: {:?}",
            p.content
        );
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "golbang");
        assert_eq!(v["limit"], 5);
    }

    #[test]
    fn parse_glm46_newline_arg_key() {
        let text = concat!(
            "<tool_call>special_function\n",
            "<arg_key>arg1</arg_key>\n<arg_value>1</arg_value>\n",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.calls[0].name, "special_function");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["arg1"], 1);
    }

    #[test]
    fn parse_glm53_parallel_calls_and_json_value() {
        let text = concat!(
            "ok\n",
            "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>",
            "<tool_call>nagar-mcp_k8s_list_pods",
            "<arg_key>namespace</arg_key><arg_value>default</arg_value>",
            "<arg_key>opts</arg_key><arg_value>{\"limit\":2}</arg_value>",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        assert_eq!(p.content, "ok");
        assert_eq!(p.calls.len(), 2);
        assert_eq!(p.calls[0].name, "bash");
        let a: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(a["command"], "ls");
        assert_eq!(p.calls[1].name, "nagar-mcp_k8s_list_pods");
        let b: Value = serde_json::from_str(&p.calls[1].arguments).unwrap();
        assert_eq!(b["namespace"], "default");
        assert_eq!(b["opts"]["limit"], 2);
    }

    #[test]
    fn parse_glm53_no_arg_call() {
        let p = parse_tool_calls("<tool_call>get_history</tool_call>");
        assert_eq!(p.calls.len(), 1);
        assert_eq!(p.calls[0].name, "get_history");
        assert_eq!(p.calls[0].arguments, "{}");
    }

    #[test]
    fn parse_glm53_korean_arg_value() {
        let text = concat!(
            "<tool_call>wiki_search",
            "<arg_key>project_name</arg_key><arg_value>golbang</arg_value>",
            "<arg_key>query</arg_key><arg_value>도구 호출</arg_value>",
            "</tool_call>"
        );
        let p = parse_tool_calls(text);
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["query"], "도구 호출");
    }

    #[test]
    fn glm_prose_mention_is_not_a_tool_call() {
        let text = "확인하겠습니다.<tool_call>` 태그가 텍스트로 그대로 출력된 채";
        let p = parse_tool_calls(text);
        assert!(
            p.calls.is_empty(),
            "prose must not become a tool: {:?}",
            p.calls
        );
        assert!(p.content.contains("확인하겠습니다."));
    }

    #[test]
    fn glm_parser_does_not_steal_qwen_function_eq() {
        let text = r#"<tool_call>
<function=get_history>
<parameter=project_name>
golbang
</parameter>
</function>
</tool_call>"#;
        let p = parse_tool_calls(text);
        assert_eq!(p.calls[0].name, "get_history");
        let v: Value = serde_json::from_str(&p.calls[0].arguments).unwrap();
        assert_eq!(v["project_name"], "golbang");
    }

    #[test]
    fn incremental_holds_glm_arg_key_then_finishes() {
        let mut p = ToolCallParser::new();
        assert_eq!(p.push("hello ").as_deref(), Some("hello "));
        assert!(p.push("<tool_call>bash").is_none());
        assert!(p
            .push("<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>")
            .is_none());
        let done = p.finish();
        assert!(done.content.is_empty());
        assert_eq!(done.calls.len(), 1);
        assert_eq!(done.calls[0].name, "bash");
        assert_eq!(done.calls[0].id, "call_1");
        let v: Value = serde_json::from_str(&done.calls[0].arguments).unwrap();
        assert_eq!(v["command"], "ls");
    }
}
