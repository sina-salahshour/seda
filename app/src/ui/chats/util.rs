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
