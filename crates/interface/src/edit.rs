//! Apply validated structured edits while preserving untouched source and comments.
use serde_json::{Map, Value};
use toml_edit::{DocumentMut, Item, TableLike};
use crate::{Config, Diagnostic, InterfaceResult, Plan, Platform};

pub fn apply_config(source: &str, candidate: Config, platform: Platform) -> InterfaceResult<String> {
    let before = Plan::new(Config::from_toml(source)?,platform)?;
    let after = Plan::new(candidate,platform)?;
    let old = serde_json::to_value(before.config).map_err(error)?;
    let new = serde_json::to_value(&after.config).map_err(error)?;
    if old == new { return Ok(source.into()); }
    let mut document = source.parse::<DocumentMut>().map_err(error)?;
    let template = after.config.to_toml()?.parse::<DocumentMut>().map_err(error)?;
    merge_table(document.as_table_mut(),template.as_table(),old.as_object().unwrap(),new.as_object().unwrap());
    let edited = document.to_string();
    // Validate the actual edited text, including representations retained from the source.
    let actual = Plan::new(Config::from_toml(&edited)?,platform)?;
    if actual.config != after.config { return Err(error("Structured edit did not preserve the requested calculation")); }
    Ok(edited)
}

fn error(error: impl std::fmt::Display) -> Diagnostic {
    Diagnostic::new("configuration_edit","",error.to_string())
}

fn merge_table(destination: &mut dyn TableLike, template: &dyn TableLike, old: &Map<String,Value>, new: &Map<String,Value>) {
    for key in old.keys() { if !new.contains_key(key) { destination.remove(key); } }
    for (key,value) in new {
        if old.get(key) == Some(value) { continue; }
        let Some(item) = template.get(key) else { continue; };
        if let Some(current) = destination.get_mut(key) {
            merge_item(current,item,old.get(key).unwrap_or(&Value::Null),value);
        } else { destination.insert(key,item.clone()); }
    }
}

fn merge_item(destination: &mut Item, template: &Item, old: &Value, new: &Value) {
    if old == new { return; }
    if let (Some(old),Some(new)) = (old.as_object(),new.as_object()) {
        if let (Some(dst),Some(src)) = (destination.as_table_like_mut(),template.as_table_like()) {
            merge_table(dst,src,old,new); return;
        }
    }
    if let (Some(old),Some(new)) = (old.as_array(),new.as_array()) {
        if let (Some(dst),Some(src)) = (destination.as_array_of_tables_mut(),template.as_array_of_tables()) {
            if old.len() == new.len() && dst.len() == src.len() && dst.len() == new.len() {
                for i in 0..new.len() {
                    if let (Some(old),Some(new)) = (old[i].as_object(),new[i].as_object()) {
                        merge_table(dst.get_mut(i).unwrap(),src.get(i).unwrap(),old,new);
                    }
                }
                return;
            }
        }
    }
    let mut replacement = template.clone();
    if let (Some(old),Some(new)) = (destination.as_value(),replacement.as_value_mut()) {
        *new.decor_mut() = old.decor().clone();
    }
    *destination = replacement;
}
