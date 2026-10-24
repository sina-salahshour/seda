//! Slash commands and skills for the composer's `/` autocomplete.
//!
//! Sources, first occurrence of a name wins:
//! `<cwd>/.claude/skills/*/SKILL.md`, `<cwd>/.claude/commands/**/*.md`,
//! `<home>/.claude/skills/*/SKILL.md`, `<home>/.claude/commands/**/*.md`, and
//! plugin skills under `<home>/.claude/plugins/` named `<plugin>:<skill>`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use super::truncate;

/// One completion: `name` without the leading `/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillInfo {
    pub name: String,
    /// First line of the front-matter `description:`; may be empty.
    pub description: String,
}

/// How deep the plugin walk goes below `~/.claude/plugins`.
const PLUGIN_DEPTH: usize = 8;
/// Directories never entered during the plugin walk.
const SKIP_DIRS: &[&str] = &[".trash", "node_modules", ".git"];
/// Longest description kept.
const DESCRIPTION_MAX: usize = 120;

/// Scan the user's and the project's skills and commands. Blocking file I/O.
pub fn scan(home: &Path, cwd: Option<&Path>) -> Vec<SkillInfo> {
    let mut found = Found::default();
    if let Some(cwd) = cwd {
        skills_in(&cwd.join(".claude/skills"), None, &mut found);
        commands_in(&cwd.join(".claude/commands"), &mut found);
    }
    skills_in(&home.join(".claude/skills"), None, &mut found);
    commands_in(&home.join(".claude/commands"), &mut found);
    let plugins = home.join(".claude/plugins");
    let mut visited = HashSet::new();
    walk_plugins(&plugins, &plugins, 0, &mut visited, &mut found);
    let mut list = found.list;
    list.sort_by(|a, b| a.name.cmp(&b.name));
    list
}

/// `scanned` plus the names in `commands` (from Claude's init event) that
/// are not already in it, sorted by name.
pub fn merge(mut scanned: Vec<SkillInfo>, commands: &[String]) -> Vec<SkillInfo> {
    for name in commands {
        let name = name.trim_start_matches('/');
        if !name.is_empty() && !scanned.iter().any(|s| s.name == name) {
            scanned.push(SkillInfo { name: name.to_string(), description: String::new() });
        }
    }
    scanned.sort_by(|a, b| a.name.cmp(&b.name));
    scanned
}

#[derive(Default)]
struct Found {
    names: HashSet<String>,
    list: Vec<SkillInfo>,
}

impl Found {
    fn add(&mut self, name: String, description: String) {
        if !name.is_empty() && self.names.insert(name.clone()) {
            self.list.push(SkillInfo { name, description });
        }
    }
}

/// Sorted entries of a directory; empty when unreadable.
fn entries(dir: &Path) -> Vec<PathBuf> {
    let Ok(read) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths: Vec<PathBuf> = read.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    paths
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `<dir>/*/SKILL.md`; names get `<prefix>:` when given.
fn skills_in(dir: &Path, prefix: Option<&str>, found: &mut Found) {
    for skill_dir in entries(dir) {
        let Ok(text) = std::fs::read_to_string(skill_dir.join("SKILL.md")) else {
            continue;
        };
        let meta = front_matter(&text);
        let name = meta.name.unwrap_or_else(|| file_name(&skill_dir));
        let name = match prefix {
            Some(p) => format!("{p}:{name}"),
            None => name,
        };
        found.add(name, meta.description);
    }
}

/// `<dir>/**/*.md`; `a/b.md` is named `a:b`.
fn commands_in(dir: &Path, found: &mut Found) {
    fn walk(dir: &Path, prefix: &str, depth: usize, found: &mut Found) {
        if depth > PLUGIN_DEPTH {
            return;
        }
        for path in entries(dir) {
            let name = file_name(&path);
            if path.is_dir() {
                if !name.starts_with('.') {
                    walk(&path, &format!("{prefix}{name}:"), depth + 1, found);
                }
                continue;
            }
            let Some(stem) = name.strip_suffix(".md") else { continue };
            let Ok(text) = std::fs::read_to_string(&path) else { continue };
            found.add(format!("{prefix}{stem}"), front_matter(&text).description);
        }
    }
    walk(dir, "", 0, found);
}

/// Find `skills/<dir>/SKILL.md` anywhere below the plugins root.
fn walk_plugins(
    root: &Path,
    dir: &Path,
    depth: usize,
    visited: &mut HashSet<PathBuf>,
    found: &mut Found,
) {
    if depth > PLUGIN_DEPTH {
        return;
    }
    // Following symlinks is fine; visiting a real directory twice is not.
    let Ok(real) = std::fs::canonicalize(dir) else { return };
    if !visited.insert(real) {
        return;
    }
    for path in entries(dir) {
        if !path.is_dir() {
            continue;
        }
        let name = file_name(&path);
        if SKIP_DIRS.contains(&name.as_str()) {
            continue;
        }
        if name == "skills" {
            let plugin = plugin_name(root, &path);
            skills_in(&path, Some(&plugin), found);
            continue;
        }
        walk_plugins(root, &path, depth + 1, visited, found);
    }
}

/// `name` of the nearest `.claude-plugin/plugin.json` walking up from the
/// `skills` directory's parent (not above `root`); else that parent's name.
fn plugin_name(root: &Path, skills: &Path) -> String {
    let parent = skills.parent().unwrap_or(root);
    let mut dir = Some(parent);
    while let Some(d) = dir {
        if !d.starts_with(root) || d == root {
            break;
        }
        if let Ok(text) = std::fs::read_to_string(d.join(".claude-plugin/plugin.json"))
            && let Ok(value) = serde_json::from_str::<serde_json::Value>(&text)
            && let Some(name) = value.get("name").and_then(|n| n.as_str())
            && !name.trim().is_empty()
        {
            return name.trim().to_string();
        }
        dir = d.parent();
    }
    file_name(parent)
}

#[derive(Default, Debug, PartialEq)]
struct Meta {
    name: Option<String>,
    description: String,
}

/// `name:` and `description:` from a leading `---` … `---` block.
fn front_matter(text: &str) -> Meta {
    let mut meta = Meta::default();
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines();
    if lines.next().map(str::trim_end) != Some("---") {
        return meta;
    }
    let block: Vec<&str> = lines.take_while(|l| l.trim_end() != "---").collect();
    for (i, line) in block.iter().enumerate() {
        // Top-level keys only.
        if line.starts_with(char::is_whitespace) {
            continue;
        }
        let Some((key, value)) = line.split_once(':') else { continue };
        let value = value.trim();
        match key.trim() {
            "name" => {
                let name = unquote(value);
                if !name.is_empty() {
                    meta.name = Some(name.to_string());
                }
            }
            "description" => {
                let first = if value.is_empty() || value.starts_with(['>', '|']) {
                    // Block scalar (or value on the next line): next indented line.
                    block[i + 1..]
                        .iter()
                        .take_while(|l| l.trim().is_empty() || l.starts_with(char::is_whitespace))
                        .map(|l| l.trim())
                        .find(|l| !l.is_empty())
                        .unwrap_or_default()
                } else {
                    value
                };
                meta.description = truncate(unquote(first).trim(), DESCRIPTION_MAX);
            }
            _ => {}
        }
    }
    meta
}

fn unquote(value: &str) -> &str {
    let value = value.trim();
    for q in ['"', '\''] {
        if let Some(inner) = value.strip_prefix(q) {
            return inner.strip_suffix(q).unwrap_or(inner);
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or_default();
            let dir = std::env::temp_dir()
                .join(format!("seda-skills-{tag}-{}-{nanos}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }

        fn write(&self, rel: &str, text: &str) {
            let path = self.0.join(rel);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, text).unwrap();
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn info(name: &str, description: &str) -> SkillInfo {
        SkillInfo { name: name.into(), description: description.into() }
    }

    #[test]
    fn scan_all_sources() {
        let home = TempDir::new("home");
        let project = TempDir::new("project");
        home.write(
            ".claude/skills/swarm/SKILL.md",
            "---\nname: swarm\ndescription: \"Run a team of agents.\"\n---\n# Swarm\n",
        );
        home.write(
            ".claude/skills/shared/SKILL.md",
            "---\nname: shared\ndescription: home version\n---\n",
        );
        home.write(".claude/skills/bare/SKILL.md", "no front matter\n");
        home.write(
            ".claude/skills/folded/SKILL.md",
            "---\ndescription: >\n  Folded first line\n  second line\nname: 'folded-skill'\n---\n",
        );
        home.write(".claude/commands/deploy.md", "---\ndescription: Ship it\n---\nbody");
        home.write(".claude/commands/git/sync.md", "Sync with remote.");
        home.write(".claude/commands/notes.txt", "ignored");
        project.write(
            ".claude/skills/shared/SKILL.md",
            "---\nname: shared\ndescription: project version\n---\n",
        );
        project.write(".claude/commands/local.md", "---\ndescription: |\n  Literal text\n---\n");
        // Plugin with plugin.json above the skills dir.
        home.write(
            ".claude/plugins/cache/market/tool/1.0/.claude-plugin/plugin.json",
            r#"{"name": "tempad-dev", "version": "1.0"}"#,
        );
        home.write(
            ".claude/plugins/cache/market/tool/1.0/skills/to-code/SKILL.md",
            "---\nname: figma-to-code\ndescription: Turn designs into code.\n---\n",
        );
        // Plugin without plugin.json: named after the skills dir's parent.
        home.write(".claude/plugins/synced/abc/figma/skills/use/SKILL.md", "---\ndescription: Use it\n---\n");
        // Deleted plugin: ignored.
        home.write(".claude/plugins/.trash/old/skills/gone/SKILL.md", "---\nname: gone\n---\n");
        home.write(".claude/plugins/x/node_modules/y/skills/nm/SKILL.md", "---\nname: nm\n---\n");

        let found = scan(&home.0, Some(&project.0));
        assert_eq!(
            found,
            vec![
                info("bare", ""),
                info("deploy", "Ship it"),
                info("figma:use", "Use it"),
                info("folded-skill", "Folded first line"),
                info("git:sync", ""),
                info("local", "Literal text"),
                info("shared", "project version"),
                info("swarm", "Run a team of agents."),
                info("tempad-dev:figma-to-code", "Turn designs into code."),
            ]
        );

        // Without a project the home version wins.
        let found = scan(&home.0, None);
        assert!(found.contains(&info("shared", "home version")));
        assert!(!found.iter().any(|s| s.name == "local"));
    }

    #[test]
    fn missing_dirs_are_empty() {
        let home = TempDir::new("empty");
        assert!(scan(&home.0, Some(&home.0.join("nope"))).is_empty());
    }

    #[test]
    fn long_description_truncated() {
        let long = "x".repeat(300);
        let meta = front_matter(&format!("---\ndescription: {long}\n---\n"));
        assert_eq!(meta.description.chars().count(), DESCRIPTION_MAX);
        assert!(meta.description.ends_with('…'));
    }

    #[test]
    fn merge_with_init_commands() {
        let scanned = vec![info("swarm", "Run a team"), info("deploy", "Ship it")];
        let commands = ["compact".to_string(), "swarm".to_string(), "/clear".to_string()];
        assert_eq!(
            merge(scanned, &commands),
            vec![
                info("clear", ""),
                info("compact", ""),
                info("deploy", "Ship it"),
                info("swarm", "Run a team"),
            ]
        );
    }
}
