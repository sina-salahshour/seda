//! Microphone capture for gpui-kit's `SpeechState`, with a selectable input
//! device (`Settings.input_device`, `None` = the host's default input, which
//! on this system is PipeWire through ALSA's `default`/`pipewire` PCM).
//!
//! Adapted from gpui-kit's built-in `Microphone`: the stream opens with the
//! device's own default config (any rate, channel count and sample format)
//! and is downmixed and resampled in software to the recognizer's format.
//! Failures are logged to stderr with the real cause and reported to the
//! `SpeechState` as `SpeechError::Input`, which the UI shows.

use std::f32::consts::TAU;

use anyhow::{Context as _, anyhow};
use gpui_kit::component::speech::{AudioFormat, AudioInput, AudioSink, SpeechError};
use gpui_kit::{App, Subscription};
use rodio::cpal::{
    self, FromSample, Sample as _, SampleFormat, SizedSample, StreamConfig,
    traits::{DeviceTrait as _, HostTrait as _, StreamTrait as _},
};

use crate::store::AppSettings;

/// Names of the input devices cpal can open (for the Settings picker).
/// Blocking (ALSA probes devices); call off the UI thread.
pub fn input_device_names() -> Vec<String> {
    let host = cpal::default_host();
    let mut names: Vec<String> = host
        .input_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default();
    names.dedup();
    names
}

fn open_device(name: Option<&str>) -> anyhow::Result<cpal::Device> {
    let host = cpal::default_host();
    if let Some(name) = name {
        if let Some(device) = host
            .input_devices()
            .context("listing input devices")?
            .find(|d| d.name().is_ok_and(|n| n == name))
        {
            return Ok(device);
        }
        eprintln!("seda: input device {name:?} not found, using the default input");
    }
    host.default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))
}

/// `AudioInput` that captures from the device chosen in the settings.
#[derive(Debug, Default, Clone)]
pub struct SettingsMicrophone;

impl AudioInput for SettingsMicrophone {
    fn start(
        &self,
        format: AudioFormat,
        sink: AudioSink,
        cx: &mut App,
    ) -> Result<Subscription, SpeechError> {
        let name = AppSettings::get(cx).input_device.clone();
        let started = start_stream(name.as_deref());
        let (stream, rx, source_rate) = match started {
            Ok(started) => started,
            Err(err) => {
                eprintln!("seda: microphone failed: {err:#}");
                return Err(SpeechError::input(err));
            }
        };

        let task = cx.spawn(async move |cx| {
            let mut converter = Converter::new(source_rate, format);
            let mut mono = Vec::new();
            while let Ok(first) = rx.recv().await {
                let mut error = None;
                for capture in
                    std::iter::once(first).chain(std::iter::from_fn(|| rx.try_recv().ok()))
                {
                    match capture {
                        Capture::Samples(samples) => mono.extend_from_slice(&samples),
                        Capture::Error(message) => error = Some(message),
                    }
                }
                let samples = converter.convert(&mono);
                mono.clear();
                cx.update(|cx| {
                    if !samples.is_empty() {
                        sink.push(samples, cx);
                    }
                    if let Some(message) = error.take() {
                        eprintln!("seda: microphone stream error: {message}");
                        sink.error(SpeechError::input(anyhow!(message)), cx);
                    }
                });
            }
        });

        Ok(Subscription::new(move || {
            drop(stream);
            drop(task);
        }))
    }
}

type Started = (cpal::Stream, async_channel::Receiver<Capture>, u32);

fn start_stream(name: Option<&str>) -> anyhow::Result<Started> {
    let device = open_device(name)?;
    let device_name = device.name().unwrap_or_else(|_| "?".into());
    let supported = device
        .default_input_config()
        .with_context(|| format!("default config of {device_name:?}"))?;
    let sample_format = supported.sample_format();
    let config: StreamConfig = supported.into();
    let source_rate = config.sample_rate.0;

    let (tx, rx) = async_channel::unbounded();
    let stream = match sample_format {
        SampleFormat::F32 => build_stream::<f32>(&device, &config, tx),
        SampleFormat::I16 => build_stream::<i16>(&device, &config, tx),
        SampleFormat::U16 => build_stream::<u16>(&device, &config, tx),
        SampleFormat::I32 => build_stream::<i32>(&device, &config, tx),
        SampleFormat::U8 => build_stream::<u8>(&device, &config, tx),
        SampleFormat::F64 => build_stream::<f64>(&device, &config, tx),
        other => Err(anyhow!("unsupported sample format {other}")),
    }
    .with_context(|| {
        format!(
            "opening {device_name:?} ({sample_format}, {} ch, {source_rate} Hz)",
            config.channels
        )
    })?;
    stream.play().context("starting the input stream")?;
    Ok((stream, rx, source_rate))
}

enum Capture {
    Samples(Vec<f32>),
    Error(String),
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    tx: async_channel::Sender<Capture>,
) -> anyhow::Result<cpal::Stream>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let channels = config.channels.max(1) as usize;
    let error_tx = tx.clone();
    Ok(device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mono = data
                .chunks(channels)
                .map(|frame| {
                    frame.iter().map(|&s| f32::from_sample(s)).sum::<f32>() / frame.len() as f32
                })
                .collect();
            _ = tx.try_send(Capture::Samples(mono));
        },
        move |error| {
            _ = error_tx.try_send(Capture::Error(error.to_string()));
        },
        None,
    )?)
}

/// Linear resampler with a low-pass for downsampling (from gpui-kit).
struct Converter {
    step: f64,
    position: f64,
    previous: f32,
    low_pass: Option<LowPass>,
    channels: usize,
}

impl Converter {
    fn new(source_rate: u32, format: AudioFormat) -> Self {
        let target_rate = format.sample_rate().max(1);
        let source_rate = source_rate.max(1);
        Self {
            step: source_rate as f64 / target_rate as f64,
            position: 1.,
            previous: 0.,
            low_pass: (source_rate > target_rate)
                .then(|| LowPass::new(target_rate as f32 * 0.45, source_rate as f32)),
            channels: format.channels().max(1) as usize,
        }
    }

    fn convert(&mut self, input: &[f32]) -> Vec<i16> {
        if input.is_empty() {
            return Vec::new();
        }
        let filtered: Vec<f32>;
        let input = match self.low_pass.as_mut() {
            Some(low_pass) => {
                filtered = input.iter().map(|&x| low_pass.process(x)).collect();
                &filtered[..]
            }
            None => input,
        };
        let previous = self.previous;
        let sample = |ix: usize| if ix == 0 { previous } else { input[ix - 1] };
        let last = input.len();
        let len = last as f64;
        let mut output = Vec::with_capacity(((len / self.step) as usize + 1) * self.channels);
        while self.position <= len {
            let ix = self.position.floor() as usize;
            let frac = (self.position - ix as f64) as f32;
            let next = sample((ix + 1).min(last));
            let value = sample(ix) * (1. - frac) + next * frac;
            let value = (value.clamp(-1., 1.) * i16::MAX as f32) as i16;
            output.extend(std::iter::repeat_n(value, self.channels));
            self.position += self.step;
        }
        self.position -= len;
        self.previous = input[last - 1];
        output
    }
}

struct LowPass {
    alpha: f32,
    stages: [f32; 2],
}

impl LowPass {
    fn new(cutoff: f32, sample_rate: f32) -> Self {
        Self {
            alpha: 1. - (-TAU * cutoff / sample_rate).exp(),
            stages: [0.; 2],
        }
    }

    fn process(&mut self, mut x: f32) -> f32 {
        for stage in &mut self.stages {
            *stage += self.alpha * (x - *stage);
            x = *stage;
        }
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converter_downsamples_48k_to_24k() {
        let mut converter = Converter::new(48_000, AudioFormat::new(24_000, 1));
        let total: usize = (0..10).map(|_| converter.convert(&[0.; 480]).len()).sum();
        assert_eq!(total, 2_400);
    }

    /// Records ~2 s from the configured default input and checks it is not silent.
    /// `cargo test --bin seda mic_records_sound -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mic_records_sound() {
        println!("inputs: {:?}", input_device_names());
        let (stream, rx, rate) = start_stream(None).expect("open default input");
        std::thread::sleep(std::time::Duration::from_secs(2));
        drop(stream);
        let mut samples = Vec::new();
        while let Ok(Capture::Samples(chunk)) = rx.try_recv() {
            samples.extend(chunk);
        }
        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt();
        println!("rate {rate} Hz, {} frames, rms {rms:.6}", samples.len());
        assert!(samples.len() as u32 > rate, "captured under 1 s");
        assert!(rms > 0.0, "silent capture");
    }
}
