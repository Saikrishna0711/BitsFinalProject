import numpy as np
import librosa


def acoustic_features(
    wav_path: str,
    sr: int = 16000,
    n_mfcc: int = 40
) -> np.ndarray:
    """
    Extract acoustic features from a WAV file.

    - Loads audio at `sr` sampling rate.
    - Computes MFCCs (n_mfcc coefficients).
    - Extracts mean pitch contour using librosa.piptrack.
    - Computes spectral flux (onset strength), then repeats to match MFCC dims.

    Returns
    -------
    feats : np.ndarray, shape (T, F)
        Time-major feature matrix where F = n_mfcc + 1 (pitch) + n_mfcc (flux).
    """
    # 1) Load and resample
    y, _ = librosa.load(wav_path, sr=sr)

    # 2) MFCCs: shape (n_mfcc, T)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    # 3) Pitch: use piptrack and take mean energy-weighted pitch
    pitch, magnitudes = librosa.pyin(y=y, sr=sr)
    # Compute weighted average per frame
    weighted_pitch = np.sum(pitch * magnitudes, axis=0) / (np.sum(magnitudes, axis=0) + 1e-6)
    weighted_pitch = weighted_pitch[np.newaxis, :]

    # 4) Spectral flux (onset strength): shape (T,)
    spec_flux = librosa.onset.onset_strength(y=y, sr=sr)
    # Expand to match MFCC channels
    spec_flux_feat = np.tile(spec_flux[np.newaxis, :], (n_mfcc, 1))

    # 5) Stack: (n_mfcc + 1 + n_mfcc, T)
    feats = np.vstack([mfcc, weighted_pitch, spec_flux_feat])

    # 6) Transpose to (T, F)
    return feats.T
