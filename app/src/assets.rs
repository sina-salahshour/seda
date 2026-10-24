//! Asset source: gpui-kit's default component icons plus the Lucide icons
//! this app uses. Use them as `gpui_kit::assets::IconName::X` with
//! `Icon::new(..)` / `Button::icon(..)`. An icon not listed here renders
//! blank: add its name to the list (full catalog: gpui-kit-assets
//! `assets/icons`, PascalCase of the file name).

use std::borrow::Cow;

use gpui_kit::{AssetSource, Result, SharedString};

gpui_kit::assets::icon_assets!(
    pub ExtraIcons,
    [
        Activity,
        AudioLines,
        AudioWaveform,
        Brain,
        ChartLine,
        CirclePause,
        CirclePlay,
        CircleStop,
        Clock,
        Download,
        Dumbbell,
        FileMusic,
        FlaskConical,
        Gauge,
        GraduationCap,
        Headphones,
        Hourglass,
        Languages,
        ListChecks,
        MessageCircle,
        MessageSquarePlus,
        MessagesSquare,
        MicOff,
        MicVocal,
        Monitor,
        Pencil,
        Radio,
        RotateCcw,
        Save,
        Send,
        SendHorizontal,
        ServerOff,
        Sparkles,
        SquarePen,
        Terminal,
        Timer,
        Trash,
        Upload,
        Volume2,
        VolumeX,
        Wrench,
        X,
        Zap,
    ]
);

pub struct AppAssets;

impl AssetSource for AppAssets {
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        if let Some(bytes) = ExtraIcons.load(path)? {
            return Ok(Some(bytes));
        }
        gpui_kit::assets::Assets.load(path)
    }

    fn list(&self, path: &str) -> Result<Vec<SharedString>> {
        let mut paths = gpui_kit::assets::Assets.list(path)?;
        paths.extend(ExtraIcons.list(path)?);
        paths.sort();
        paths.dedup();
        Ok(paths)
    }
}
