use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};

/// Write interleaved 16-bit PCM as a WAV file.
pub fn write_wav(path: &Path, samples: &[i16], sample_rate: u32, channels: u16) -> Result<()> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("cannot create {}", path.display()))?;
    for &sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(())
}

/// Encode interleaved 16-bit PCM as complete WAV file bytes.
#[allow(dead_code)]
pub(crate) fn wav_bytes(samples: &[i16], sample_rate: u32, channels: u16) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for &sample in samples {
            writer.write_sample(sample)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// A fresh path like `/tmp/seda/<prefix>-<id>.wav`.
pub fn temp_wav_path(prefix: &str) -> PathBuf {
    crate::store::temp_dir().join(format!("{prefix}-{}.wav", crate::store::new_id()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let path = temp_wav_path("test");
        let samples: Vec<i16> = (0..2400)
            .map(|i| ((i % 100) * 300 - 15000) as i16)
            .collect();
        write_wav(&path, &samples, 24_000, 1).unwrap();
        let mut reader = hound::WavReader::open(&path).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 24_000);
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.bits_per_sample, 16);
        let read: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read, samples);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn bytes_round_trip() {
        let samples = vec![1i16, -2, 3, -4];
        let bytes = wav_bytes(&samples, 16_000, 2).unwrap();
        let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes)).unwrap();
        assert_eq!(reader.spec().channels, 2);
        let read: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(read, samples);
    }

    #[test]
    fn bad_path_is_error() {
        assert!(write_wav(Path::new("/nonexistent-dir/x.wav"), &[0], 16_000, 1).is_err());
    }
}
