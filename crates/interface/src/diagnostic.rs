use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct Diagnostic {
    pub code: String,
    pub path: String,
    pub message: String,
    /// Half-open UTF-8 byte offsets in the original TOML document.
    pub span: Option<[usize; 2]>,
}

impl Diagnostic {
    pub fn new(code: &str, path: impl Into<String>, message: impl Into<String>) -> Self {
        Self { code: code.into(), path: path.into(), message: message.into(), span: None }
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.path.is_empty() { write!(f, "{}", self.message) }
        else { write!(f, "{}: {}", self.path, self.message) }
    }
}

impl std::error::Error for Diagnostic {}

pub type InterfaceResult<T> = Result<T, Diagnostic>;

