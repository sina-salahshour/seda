//! Small helpers for the chats page: select items and time labels.

use gpui_kit::SharedString;
use gpui_kit::component::select::{SearchableVec, SelectItem};

use crate::sidecar::Profile;
use crate::ui::common::fa_digits;

/// One option of a `Select`: a stable value and its shown title.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    value: String,
    title: SharedString,
}

impl Choice {
    pub fn new(value: impl Into<String>, title: impl Into<SharedString>) -> Self {
        Self {
            value: value.into(),
            title: title.into(),
        }
    }
}

impl SelectItem for Choice {
    type Value = String;

    fn title(&self) -> SharedString {
        self.title.clone()
    }

    fn value(&self) -> &String {
        &self.value
    }
}

pub type Choices = SearchableVec<Choice>;

/// The value used for "no voice" in voice selects.
pub const NO_VOICE: &str = "";

/// Voice options: "no voice" first, then every profile. A `current` id that
/// is not among the profiles (sidecar offline, profile deleted) stays
/// selectable so the session's choice is not silently dropped.
pub fn voice_choices(profiles: &[Profile], current: Option<&str>) -> Vec<Choice> {
    let mut choices = vec![Choice::new(NO_VOICE, "بدون صدا")];
    choices.extend(
        profiles
            .iter()
            .map(|profile| Choice::new(profile.id.clone(), profile.name.clone())),
    );
    if let Some(current) = current.filter(|id| !id.is_empty())
        && !profiles.iter().any(|profile| profile.id == current)
    {
        choices.push(Choice::new(current, format!("صدای {current}")));
    }
    choices
}

/// Model options; keeps `current` even when Ollama does not list it (now).
pub fn model_choices(models: &[String], current: Option<&str>) -> Vec<Choice> {
    let mut choices: Vec<Choice> = models
        .iter()
        .map(|model| Choice::new(model.clone(), model.clone()))
        .collect();
    if let Some(current) = current
        && !models.iter().any(|model| model == current)
    {
        choices.push(Choice::new(current, current.to_string()));
    }
    choices
}

/// A short Persian label for how long ago `rfc3339` was.
pub fn relative_time(rfc3339: &str) -> String {
    let Ok(then) = chrono::DateTime::parse_from_rfc3339(rfc3339) else {
        return String::new();
    };
    let then = then.with_timezone(&chrono::Utc);
    let seconds = (chrono::Utc::now() - then).num_seconds().max(0);
    let label = match seconds {
        0..60 => return "همین حالا".into(),
        60..3_600 => format!("{} دقیقه پیش", seconds / 60),
        3_600..86_400 => format!("{} ساعت پیش", seconds / 3_600),
        86_400..172_800 => return "دیروز".into(),
        172_800..604_800 => format!("{} روز پیش", seconds / 86_400),
        _ => then.format("%Y-%m-%d").to_string(),
    };
    format!("\u{200F}{}", fa_digits(&label))
}

/// First `max` characters of `text`'s first line, with an ellipsis when cut.
pub fn one_line(text: &str, max: usize) -> String {
    let line = text.lines().next().unwrap_or_default().trim();
    let mut out: String = line.chars().take(max).collect();
    if line.chars().count() > max || text.lines().nth(1).is_some() {
        out.push('…');
    }
    out
}

/// The value used for "the backend's default" in model and reasoning selects.
pub const DEFAULT_CHOICE: &str = "";

/// `""` → `None`, anything else → `Some`.
pub fn non_default(value: &str) -> Option<String> {
    (value != DEFAULT_CHOICE).then(|| value.to_string())
}

/// Claude Code `--model` options: the CLI default, the aliases, and the
/// session's current value if it is something else (a full model name).
pub fn claude_model_choices(current: Option<&str>) -> Vec<Choice> {
    let mut choices = vec![Choice::new(DEFAULT_CHOICE, "مدل پیش‌فرض")];
    choices.extend(
        ["fable", "opus", "sonnet", "haiku"]
            .into_iter()
            .map(|alias| Choice::new(alias, alias)),
    );
    if let Some(current) = current.filter(|c| !c.is_empty())
        && !choices.iter().any(|c| c.value == current)
    {
        choices.push(Choice::new(current, current.to_string()));
    }
    choices
}

/// Model options with a leading "default" entry (opencode, where `-m` is optional).
pub fn optional_model_choices(models: &[String], current: Option<&str>) -> Vec<Choice> {
    let mut choices = vec![Choice::new(DEFAULT_CHOICE, "مدل پیش‌فرض")];
    choices.extend(model_choices(models, current.filter(|c| !c.is_empty())));
    choices
}

/// Which reasoning scale a backend uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningScale {
    /// `--effort`: default, low, medium, high, xhigh, max.
    Claude,
    /// `think`: off, on, low, medium, high.
    Ollama,
    /// `reasoning_effort`: default, low, medium, high.
    OpenAi,
    /// `--variant`: default, low, medium, high, max.
    OpenCode,
}

fn level_label(value: &str) -> &'static str {
    match value {
        "" => "فکر پیش‌فرض",
        "off" => "بدون فکر",
        "on" => "با فکر",
        "low" => "فکر کم",
        "medium" => "فکر متوسط",
        "high" => "فکر زیاد",
        "xhigh" => "فکر خیلی زیاد",
        "max" => "بیشترین فکر",
        _ => "",
    }
}

/// Reasoning options for `scale`; keeps an unusual `current` selectable.
pub fn reasoning_choices(scale: ReasoningScale, current: Option<&str>) -> Vec<Choice> {
    let values: &[&str] = match scale {
        ReasoningScale::Claude => &["", "low", "medium", "high", "xhigh", "max"],
        ReasoningScale::Ollama => &["off", "on", "low", "medium", "high"],
        ReasoningScale::OpenAi => &["", "low", "medium", "high"],
        ReasoningScale::OpenCode => &["", "low", "medium", "high", "max"],
    };
    let mut choices: Vec<Choice> = values
        .iter()
        .map(|v| Choice::new(*v, level_label(v)))
        .collect();
    if let Some(current) = current.filter(|c| !c.is_empty())
        && !values.contains(&current)
    {
        choices.push(Choice::new(current, format!("فکر: {current}")));
    }
    choices
}

/// The select value for a stored reasoning setting (`None` → default / off).
pub fn reasoning_value(scale: ReasoningScale, current: Option<&str>) -> String {
    match (scale, current) {
        (ReasoningScale::Ollama, None | Some("")) => "off".into(),
        (_, Some(value)) => value.to_string(),
        (_, None) => DEFAULT_CHOICE.into(),
    }
}

/// The stored reasoning setting for a select value.
pub fn reasoning_setting(scale: ReasoningScale, value: &str) -> Option<String> {
    match (scale, value) {
        (ReasoningScale::Ollama, "off") | (_, "") => None,
        (_, value) => Some(value.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_round_trip() {
        for scale in [
            ReasoningScale::Claude,
            ReasoningScale::Ollama,
            ReasoningScale::OpenAi,
            ReasoningScale::OpenCode,
        ] {
            for choice in reasoning_choices(scale, None) {
                let setting = reasoning_setting(scale, &choice.value);
                assert_eq!(reasoning_value(scale, setting.as_deref()), choice.value);
            }
        }
        assert_eq!(reasoning_value(ReasoningScale::Ollama, None), "off");
        assert_eq!(reasoning_choices(ReasoningScale::OpenAi, Some("minimal")).len(), 5);
        assert_eq!(claude_model_choices(Some("claude-opus-5-5")).len(), 6);
        assert_eq!(claude_model_choices(Some("opus")).len(), 5);
    }
}
