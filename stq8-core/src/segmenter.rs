//! Energy-based syllable segmentation.
//!
//! Segments a PTT utterance into individual syllable regions using
//! RMS energy tracking with noise floor estimation, onset/offset
//! detection, gap merging, and minimum duration filtering.

/// Analysis window size in samples (10ms at 16kHz).
const WINDOW_SIZE: usize = 160;

/// A detected syllable: start and end sample indices within the input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyllableBounds {
    pub start: usize,
    pub end: usize,
}

/// Configuration for the segmenter.
#[derive(Debug, Clone)]
pub struct SegmenterConfig {
    /// RMS energy threshold relative to noise floor (multiplier).
    pub onset_threshold: f32,
    /// Minimum syllable duration in samples.
    pub min_duration: usize,
    /// Minimum silence gap between syllables in samples.
    pub min_gap: usize,
    /// Number of frames to estimate initial noise floor.
    pub noise_floor_frames: usize,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            onset_threshold: 3.0,
            min_duration: 800, // 50ms at 16kHz
            min_gap: 400,      // 25ms at 16kHz
            noise_floor_frames: 5,
        }
    }
}

/// Segment an audio signal into syllable regions based on energy.
///
/// Assumes leading silence: the first `noise_floor_frames` frames are used to
/// estimate the ambient noise level. If the signal is active from the very start,
/// the noise floor will be high and detection may fail. Trailing samples that
/// don't fill a complete analysis frame (160 samples) are not analyzed.
///
/// Algorithm:
/// 1. Compute RMS energy per 160-sample (10ms) frame.
/// 2. Estimate noise floor from the minimum RMS of the first
///    `noise_floor_frames` frames (with a small floor to prevent division by zero).
/// 3. Mark frames where RMS exceeds `noise_floor * onset_threshold` as active.
/// 4. Find contiguous runs of active frames and convert to sample indices.
/// 5. Merge candidate regions separated by fewer than `min_gap` samples.
/// 6. Reject candidates shorter than `min_duration` samples.
pub fn segment(samples: &[f32], config: &SegmenterConfig) -> Vec<SyllableBounds> {
    if samples.is_empty() {
        return Vec::new();
    }

    // Step 1: Compute RMS energy per frame.
    let num_frames = samples.len() / WINDOW_SIZE;
    if num_frames == 0 {
        return Vec::new();
    }

    let rms_values: Vec<f32> = (0..num_frames)
        .map(|i| {
            let start = i * WINDOW_SIZE;
            let end = start + WINDOW_SIZE;
            let sum_sq: f32 = samples[start..end].iter().map(|&x| x * x).sum();
            (sum_sq / WINDOW_SIZE as f32).sqrt()
        })
        .collect();

    // Step 2: Estimate noise floor from first noise_floor_frames frames.
    let floor_count = config.noise_floor_frames.min(num_frames);
    let noise_floor = rms_values[..floor_count]
        .iter()
        .copied()
        .fold(f32::INFINITY, f32::min)
        .max(1e-6);

    // Step 3: Mark active frames.
    // Use the relative threshold, but enforce an absolute minimum so that
    // if speech starts immediately (no leading silence), the estimated
    // noise floor doesn't suppress all detection.
    let abs_min_threshold = 0.01_f32;
    let threshold = (noise_floor * config.onset_threshold).max(abs_min_threshold);
    let active: Vec<bool> = rms_values.iter().map(|&rms| rms > threshold).collect();

    // Step 4: Find contiguous runs of active frames -> candidate regions.
    let mut candidates: Vec<SyllableBounds> = Vec::new();
    let mut in_region = false;
    let mut region_start = 0;

    for (i, &is_active) in active.iter().enumerate() {
        if is_active && !in_region {
            region_start = i * WINDOW_SIZE;
            in_region = true;
        } else if !is_active && in_region {
            let region_end = i * WINDOW_SIZE;
            candidates.push(SyllableBounds {
                start: region_start,
                end: region_end,
            });
            in_region = false;
        }
    }
    // Close any region that extends to the last frame.
    if in_region {
        let region_end = num_frames * WINDOW_SIZE;
        candidates.push(SyllableBounds {
            start: region_start,
            end: region_end,
        });
    }

    if candidates.is_empty() {
        return Vec::new();
    }

    // Step 5: Merge candidates separated by fewer than min_gap samples.
    let mut merged: Vec<SyllableBounds> = Vec::new();
    merged.push(candidates[0].clone());

    for candidate in &candidates[1..] {
        let last = merged.last_mut().unwrap();
        if candidate.start - last.end < config.min_gap {
            last.end = candidate.end;
        } else {
            merged.push(candidate.clone());
        }
    }

    // Step 6: Reject candidates shorter than min_duration samples.
    merged
        .into_iter()
        .filter(|b| b.end - b.start >= config.min_duration)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;

    /// Create synthetic PCM with sine bursts at specified sample ranges.
    fn make_bursts(bursts: &[(usize, usize)], total_len: usize) -> Vec<f32> {
        let mut signal = vec![0.0f32; total_len];
        for &(start, end) in bursts {
            for i in start..end.min(total_len) {
                let t = i as f32 / SR as f32;
                signal[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.8;
            }
        }
        signal
    }

    #[test]
    fn silence_produces_no_syllables() {
        // 500ms of silence at 16kHz = 8000 samples
        let silence = vec![0.0f32; (SR as usize) / 2];
        let config = SegmenterConfig::default();
        let result = segment(&silence, &config);
        assert!(
            result.is_empty(),
            "pure silence should produce no syllables, got {result:?}"
        );
    }

    #[test]
    fn single_burst_one_syllable() {
        // 100ms burst at 16kHz = 1600 samples, starting at sample 1600
        let burst_start = 1600;
        let burst_end = 3200;
        let total = SR as usize; // 1 second
        let signal = make_bursts(&[(burst_start, burst_end)], total);
        let config = SegmenterConfig::default();
        let result = segment(&signal, &config);

        assert_eq!(
            result.len(),
            1,
            "should detect exactly 1 syllable, got {result:?}"
        );

        // Bounds should approximately match the burst region.
        // Allow tolerance of one analysis window (160 samples) on each side.
        let tol = WINDOW_SIZE;
        assert!(
            result[0].start <= burst_start + tol,
            "start {} should be near burst start {burst_start}",
            result[0].start
        );
        assert!(
            result[0].end >= burst_end - tol,
            "end {} should be near burst end {burst_end}",
            result[0].end
        );
    }

    #[test]
    fn two_bursts_with_gap() {
        // Two 100ms bursts with 100ms gap between them.
        // Burst 1: samples 1600..3200, Burst 2: samples 4800..6400
        let burst1 = (1600, 3200);
        let burst2 = (4800, 6400);
        let total = SR as usize;
        let signal = make_bursts(&[burst1, burst2], total);
        let config = SegmenterConfig::default();
        let result = segment(&signal, &config);

        assert_eq!(
            result.len(),
            2,
            "should detect 2 syllables with 100ms gap, got {result:?}"
        );

        // They should not overlap.
        assert!(
            result[0].end <= result[1].start,
            "syllables should not overlap: first ends at {}, second starts at {}",
            result[0].end,
            result[1].start
        );
    }

    #[test]
    fn very_short_burst_rejected() {
        // 20ms burst = 320 samples, which is below the default min_duration of 800.
        let burst_start = 1600;
        let burst_end = 1920; // 320 samples
        let total = SR as usize;
        let signal = make_bursts(&[(burst_start, burst_end)], total);
        let config = SegmenterConfig::default();
        let result = segment(&signal, &config);

        assert!(
            result.is_empty(),
            "20ms burst (320 samples) should be rejected by min_duration (800), got {result:?}"
        );
    }

    #[test]
    fn bursts_without_gap_merged() {
        // Two bursts with only 10ms (160 samples) gap, which is < min_gap (400).
        // Burst 1: samples 1600..3200, Burst 2: samples 3360..4960
        // Gap = 3360 - 3200 = 160 samples < 400
        let burst1 = (1600, 3200);
        let burst2 = (3360, 4960);
        let total = SR as usize;
        let signal = make_bursts(&[burst1, burst2], total);
        let config = SegmenterConfig::default();
        let result = segment(&signal, &config);

        assert_eq!(
            result.len(),
            1,
            "bursts with 10ms gap should merge into 1 syllable, got {result:?}"
        );
    }

    #[test]
    fn empty_input() {
        let config = SegmenterConfig::default();
        let result = segment(&[], &config);
        assert!(
            result.is_empty(),
            "empty input should produce no syllables, got {result:?}"
        );
    }

    #[test]
    fn bounds_within_input() {
        // Multiple bursts, verify all bounds are valid.
        let bursts = vec![(800, 2400), (4000, 5600), (7200, 8800)];
        let total = SR as usize; // 16000 samples
        let signal = make_bursts(&bursts, total);
        let config = SegmenterConfig::default();
        let result = segment(&signal, &config);

        for (i, bounds) in result.iter().enumerate() {
            assert!(
                bounds.start < bounds.end,
                "syllable {i}: start ({}) must be < end ({})",
                bounds.start,
                bounds.end
            );
            assert!(
                bounds.end <= signal.len(),
                "syllable {i}: end ({}) must be <= signal length ({})",
                bounds.end,
                signal.len()
            );
        }
    }

    #[test]
    fn default_config_values() {
        let config = SegmenterConfig::default();
        assert!((config.onset_threshold - 3.0).abs() < f32::EPSILON);
        assert_eq!(config.min_duration, 800);
        assert_eq!(config.min_gap, 400);
        assert_eq!(config.noise_floor_frames, 5);
    }

    #[test]
    fn signal_shorter_than_noise_floor_frames() {
        // Signal with only 2 frames worth of data (320 samples) but a burst,
        // ensuring noise floor estimation falls back to fewer frames.
        // With only 2 frames, both being burst, noise floor = burst energy,
        // so nothing should be detected as "above threshold" (or it's all active).
        // Use a signal with 1 silent frame + 1 active frame.
        // 3 frames total; fill frames 1-2 (index 160..480) with a tone.
        let mut signal = vec![0.0f32; WINDOW_SIZE * 3];
        for i in WINDOW_SIZE..WINDOW_SIZE * 3 {
            let t = i as f32 / SR as f32;
            signal[i] = (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.8;
        }
        let config = SegmenterConfig {
            noise_floor_frames: 10, // More than available frames
            min_duration: 0,        // Accept any length
            ..SegmenterConfig::default()
        };
        let result = segment(&signal, &config);
        // Should detect the active region (frames 1-2).
        assert!(
            !result.is_empty(),
            "should detect syllable even when noise_floor_frames > available frames"
        );
    }
}
