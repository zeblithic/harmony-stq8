//! MFCC (Mel-Frequency Cepstral Coefficient) feature extraction.
//!
//! Extracts 52-dimensional feature vectors from audio syllable segments:
//! 13 MFCCs x mean/variance x static/delta coefficients.
//!
//! Pipeline: pre-emphasis -> Hamming window -> FFT -> power spectrum
//! -> mel filterbank -> log compression -> DCT type-II -> statistics.

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::f32::consts::PI;
use std::sync::Arc;

/// Number of MFCC coefficients per frame.
pub const NUM_MFCC: usize = 13;

/// Number of mel-scale triangular filters.
pub const NUM_MEL_FILTERS: usize = 26;

/// FFT size (zero-padded frame length).
pub const FFT_SIZE: usize = 512;

/// Audio sample rate in Hz.
pub const SAMPLE_RATE: u32 = 16_000;

/// Frame length in samples (25ms at 16kHz).
pub const FRAME_LEN: usize = 400;

/// Hop length in samples (10ms at 16kHz).
pub const HOP_LEN: usize = 160;

/// Output feature dimension: 13 MFCCs x mean/var x static/delta = 52.
pub const FEATURE_DIM: usize = NUM_MFCC * 2 * 2;

/// Apply pre-emphasis filter to boost high frequencies.
///
/// y[0] = signal[0], y[n] = signal[n] - coeff * signal[n-1]
pub fn pre_emphasis(signal: &[f32], coeff: f32) -> Vec<f32> {
    if signal.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(signal.len());
    out.push(signal[0]);
    for n in 1..signal.len() {
        out.push(signal[n] - coeff * signal[n - 1]);
    }
    out
}

/// Generate a Hamming window of the given length.
///
/// w[n] = 0.54 - 0.46 * cos(2*pi*n / (len - 1))
pub fn hamming_window(len: usize) -> Vec<f32> {
    if len <= 1 {
        return vec![1.0; len];
    }
    (0..len)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f32 / (len - 1) as f32).cos())
        .collect()
}

/// Convert frequency in Hz to mel scale.
pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Convert mel scale value back to Hz.
pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

/// Create a mel-scale triangular filterbank.
///
/// Returns `num_filters` triangular filters spanning 300Hz-8kHz on the mel scale.
/// Each filter is a `Vec<f32>` of length `fft_size / 2 + 1` (power spectrum bins).
pub fn mel_filterbank(num_filters: usize, fft_size: usize, sample_rate: u32) -> Vec<Vec<f32>> {
    let num_bins = fft_size / 2 + 1;
    let low_freq = 300.0_f32;
    let high_freq = (sample_rate as f32 / 2.0).min(8000.0);

    let low_mel = hz_to_mel(low_freq);
    let high_mel = hz_to_mel(high_freq);

    // num_filters + 2 points: includes the boundary points
    let num_points = num_filters + 2;
    let mel_points: Vec<f32> = (0..num_points)
        .map(|i| low_mel + (high_mel - low_mel) * i as f32 / (num_points - 1) as f32)
        .collect();

    // Convert mel points to FFT bin indices
    let bin_points: Vec<f32> = mel_points
        .iter()
        .map(|&m| mel_to_hz(m) * fft_size as f32 / sample_rate as f32)
        .collect();

    let mut filterbank = Vec::with_capacity(num_filters);
    for i in 0..num_filters {
        let mut filter = vec![0.0_f32; num_bins];
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for (k, weight) in filter.iter_mut().enumerate() {
            let k_f = k as f32;
            if k_f > left && k_f <= center && center > left {
                *weight = (k_f - left) / (center - left);
            } else if k_f > center && k_f < right && right > center {
                *weight = (right - k_f) / (right - center);
            }
        }
        filterbank.push(filter);
    }
    filterbank
}

/// Cached resources for MFCC extraction, avoiding per-frame recomputation
/// of FFT plan, mel filterbank, and Hamming window.
struct FrameProcessor {
    fft: Arc<dyn Fft<f32>>,
    filters: Vec<Vec<f32>>,
    window: Vec<f32>,
}

impl FrameProcessor {
    fn new() -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let filters = mel_filterbank(NUM_MEL_FILTERS, FFT_SIZE, SAMPLE_RATE);
        let window = hamming_window(FRAME_LEN);
        Self {
            fft,
            filters,
            window,
        }
    }

    fn process_frame(&self, frame: &[f32]) -> Vec<f32> {
        // Pre-emphasis
        let emphasized = pre_emphasis(frame, 0.97);

        // Hamming window
        let windowed: Vec<f32> = emphasized
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect();

        // Zero-pad to FFT_SIZE and prepare complex input
        let mut fft_input: Vec<Complex<f32>> =
            windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();
        fft_input.resize(FFT_SIZE, Complex::new(0.0, 0.0));

        // FFT
        self.fft.process(&mut fft_input);

        // Power spectrum: |X[k]|^2, only first FFT_SIZE/2 + 1 bins
        let num_bins = FFT_SIZE / 2 + 1;
        let power_spectrum: Vec<f32> =
            fft_input[..num_bins].iter().map(|c| c.norm_sqr()).collect();

        // Apply filterbank and log compression
        let mel_energies: Vec<f32> = self
            .filters
            .iter()
            .map(|filter| {
                let energy: f32 = filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                energy.max(1e-10).ln()
            })
            .collect();

        // DCT type-II: extract first NUM_MFCC coefficients
        let n = mel_energies.len();
        let mut mfccs = Vec::with_capacity(NUM_MFCC);
        for k in 0..NUM_MFCC {
            let coeff: f32 = mel_energies
                .iter()
                .enumerate()
                .map(|(i, &x)| x * (PI * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32).cos())
                .sum();
            mfccs.push(coeff);
        }

        mfccs
    }
}

/// Extract MFCCs from a single audio frame.
///
/// Input: `frame` of `FRAME_LEN` samples.
/// Output: `NUM_MFCC` (13) MFCC coefficients.
///
/// Pipeline: pre-emphasize (0.97) -> Hamming window -> zero-pad to FFT_SIZE
/// -> FFT -> power spectrum -> mel filterbank -> log (floor 1e-10)
/// -> DCT type-II (first 13 coefficients).
///
/// For processing multiple frames, prefer `extract_features` which caches
/// the FFT plan and mel filterbank across frames.
pub fn extract_frame_mfccs(frame: &[f32]) -> Vec<f32> {
    let processor = FrameProcessor::new();
    processor.process_frame(frame)
}

/// Extract a 52-dimensional feature vector from a syllable audio signal.
///
/// Splits the signal into overlapping frames, extracts MFCCs per frame,
/// computes delta coefficients, then summarizes with mean and variance
/// of both static and delta MFCCs.
///
/// Output: 52 floats = 13 static means + 13 static variances
///                    + 13 delta means + 13 delta variances
pub fn extract_features(signal: &[f32]) -> Vec<f32> {
    // Ensure we have at least one frame's worth of signal
    let padded: Vec<f32> = if signal.len() < FRAME_LEN {
        let mut p = signal.to_vec();
        p.resize(FRAME_LEN, 0.0);
        p
    } else {
        signal.to_vec()
    };

    // Cache FFT plan, filterbank, and window across frames
    let processor = FrameProcessor::new();

    // Split into overlapping frames
    let mut frames_mfccs: Vec<Vec<f32>> = Vec::new();
    let mut start = 0;
    while start + FRAME_LEN <= padded.len() {
        let frame = &padded[start..start + FRAME_LEN];
        frames_mfccs.push(processor.process_frame(frame));
        start += HOP_LEN;
    }

    let num_frames = frames_mfccs.len();

    // Compute delta coefficients (first difference between adjacent frames)
    let deltas: Vec<Vec<f32>> = if num_frames <= 1 {
        vec![vec![0.0; NUM_MFCC]; num_frames]
    } else {
        (0..num_frames)
            .map(|t| {
                (0..NUM_MFCC)
                    .map(|k| {
                        if t == 0 {
                            frames_mfccs[1][k] - frames_mfccs[0][k]
                        } else if t == num_frames - 1 {
                            frames_mfccs[t][k] - frames_mfccs[t - 1][k]
                        } else {
                            (frames_mfccs[t + 1][k] - frames_mfccs[t - 1][k]) / 2.0
                        }
                    })
                    .collect()
            })
            .collect()
    };

    // Compute mean and variance for static MFCCs
    let n = num_frames as f32;
    let mut static_mean = vec![0.0_f32; NUM_MFCC];
    for frame in &frames_mfccs {
        for (k, &val) in frame.iter().enumerate() {
            static_mean[k] += val;
        }
    }
    for v in &mut static_mean {
        *v /= n;
    }

    let mut static_var = vec![0.0_f32; NUM_MFCC];
    for frame in &frames_mfccs {
        for (k, &val) in frame.iter().enumerate() {
            let diff = val - static_mean[k];
            static_var[k] += diff * diff;
        }
    }
    for v in &mut static_var {
        *v /= n;
    }

    // Compute mean and variance for delta MFCCs
    let mut delta_mean = vec![0.0_f32; NUM_MFCC];
    for frame in &deltas {
        for (k, &val) in frame.iter().enumerate() {
            delta_mean[k] += val;
        }
    }
    for v in &mut delta_mean {
        *v /= n;
    }

    let mut delta_var = vec![0.0_f32; NUM_MFCC];
    for frame in &deltas {
        for (k, &val) in frame.iter().enumerate() {
            let diff = val - delta_mean[k];
            delta_var[k] += diff * diff;
        }
    }
    for v in &mut delta_var {
        *v /= n;
    }

    // Concatenate: static_mean(13) + static_var(13) + delta_mean(13) + delta_var(13) = 52
    let mut features = Vec::with_capacity(FEATURE_DIM);
    features.extend_from_slice(&static_mean);
    features.extend_from_slice(&static_var);
    features.extend_from_slice(&delta_mean);
    features.extend_from_slice(&delta_var);
    features
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a sine wave at a given frequency.
    fn sine_wave(freq: f32, duration_samples: usize, sample_rate: u32) -> Vec<f32> {
        (0..duration_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn pre_emphasis_boosts_high_freq() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = pre_emphasis(&signal, 0.97);

        assert_eq!(y[0], 1.0, "y[0] should equal x[0]");
        assert!(
            (y[1] - (2.0 - 0.97 * 1.0)).abs() < 1e-6,
            "y[1] = x[1] - 0.97 * x[0]"
        );
        assert!(
            (y[2] - (3.0 - 0.97 * 2.0)).abs() < 1e-6,
            "y[2] = x[2] - 0.97 * x[1]"
        );
    }

    #[test]
    fn pre_emphasis_empty() {
        let y = pre_emphasis(&[], 0.97);
        assert!(y.is_empty());
    }

    #[test]
    fn hamming_window_shape() {
        let w = hamming_window(400);
        assert_eq!(w.len(), 400);

        // Endpoints should be near 0.08 (0.54 - 0.46 = 0.08)
        assert!(
            (w[0] - 0.08).abs() < 0.01,
            "first sample should be near 0.08, got {}",
            w[0]
        );
        assert!(
            (w[399] - 0.08).abs() < 0.01,
            "last sample should be near 0.08, got {}",
            w[399]
        );

        // Middle should be near 1.0
        assert!(
            (w[200] - 1.0).abs() < 0.01,
            "middle sample should be near 1.0, got {}",
            w[200]
        );

        // Symmetric
        for i in 0..200 {
            assert!(
                (w[i] - w[399 - i]).abs() < 1e-6,
                "window should be symmetric at index {i}"
            );
        }
    }

    #[test]
    fn mel_hz_roundtrip() {
        let test_freqs = [0.0, 300.0, 1000.0, 4000.0, 8000.0];
        for &hz in &test_freqs {
            let mel = hz_to_mel(hz);
            let roundtrip = mel_to_hz(mel);
            assert!(
                (roundtrip - hz).abs() < 0.1,
                "roundtrip failed for {hz}Hz: got {roundtrip}Hz"
            );
        }
    }

    #[test]
    fn mel_scale_is_monotonic() {
        let freqs = [0.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
        let mels: Vec<f32> = freqs.iter().map(|&f| hz_to_mel(f)).collect();
        for i in 1..mels.len() {
            assert!(
                mels[i] > mels[i - 1],
                "mel scale should be monotonic: mel({}) = {}, mel({}) = {}",
                freqs[i - 1],
                mels[i - 1],
                freqs[i],
                mels[i]
            );
        }
    }

    #[test]
    fn mel_filterbank_shape() {
        let fb = mel_filterbank(NUM_MEL_FILTERS, FFT_SIZE, SAMPLE_RATE);

        // Correct number of filters
        assert_eq!(fb.len(), NUM_MEL_FILTERS);

        // Each filter has correct number of bins
        let expected_bins = FFT_SIZE / 2 + 1;
        for (i, filter) in fb.iter().enumerate() {
            assert_eq!(
                filter.len(),
                expected_bins,
                "filter {i} should have {expected_bins} bins"
            );
        }

        // All weights non-negative
        for (i, filter) in fb.iter().enumerate() {
            for (k, &w) in filter.iter().enumerate() {
                assert!(w >= 0.0, "filter {i} bin {k} has negative weight: {w}");
            }
        }

        // Each filter should have at least some non-zero values
        for (i, filter) in fb.iter().enumerate() {
            let sum: f32 = filter.iter().sum();
            assert!(sum > 0.0, "filter {i} is all zeros");
        }
    }

    #[test]
    fn extract_single_frame_deterministic() {
        let frame = sine_wave(440.0, FRAME_LEN, SAMPLE_RATE);

        let mfccs1 = extract_frame_mfccs(&frame);
        let mfccs2 = extract_frame_mfccs(&frame);

        assert_eq!(mfccs1.len(), NUM_MFCC, "should produce {NUM_MFCC} MFCCs");
        assert_eq!(mfccs1, mfccs2, "same input should produce identical MFCCs");
    }

    #[test]
    fn extract_features_output_dim() {
        let signal = sine_wave(440.0, SAMPLE_RATE as usize / 2, SAMPLE_RATE);
        let features = extract_features(&signal);
        assert_eq!(
            features.len(),
            FEATURE_DIM,
            "feature vector should have {FEATURE_DIM} dimensions"
        );
    }

    #[test]
    fn extract_features_short_signal() {
        // Signal shorter than one frame should be zero-padded and still produce valid output
        let short_signal = sine_wave(440.0, 100, SAMPLE_RATE);
        let features = extract_features(&short_signal);
        assert_eq!(features.len(), FEATURE_DIM);
        for (i, &f) in features.iter().enumerate() {
            assert!(f.is_finite(), "feature {i} is not finite: {f}");
        }
    }

    #[test]
    fn silence_produces_low_energy_features() {
        let silence = vec![0.0_f32; SAMPLE_RATE as usize / 2];
        let features = extract_features(&silence);

        assert_eq!(features.len(), FEATURE_DIM);
        for (i, &f) in features.iter().enumerate() {
            assert!(
                f.is_finite(),
                "feature {i} is NaN or inf from silent input: {f}"
            );
        }
    }

    #[test]
    fn different_frequencies_produce_different_features() {
        let low = sine_wave(200.0, SAMPLE_RATE as usize / 2, SAMPLE_RATE);
        let high = sine_wave(4000.0, SAMPLE_RATE as usize / 2, SAMPLE_RATE);

        let feat_low = extract_features(&low);
        let feat_high = extract_features(&high);

        // L2 distance between feature vectors
        let l2: f32 = feat_low
            .iter()
            .zip(feat_high.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        assert!(
            l2 > 1.0,
            "200Hz and 4000Hz should produce meaningfully different features, L2 = {l2}"
        );
    }
}
