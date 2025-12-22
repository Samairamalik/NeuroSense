import numpy as np
import joblib
import librosa
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_audio(wav_path, target_sr=16000):
    """Load audio, resample to 16kHz, convert to mono, trim silence"""
    try:
        # Load audio
        y, sr = librosa.load(wav_path, sr=None, mono=False)
        
        # Convert to mono if stereo
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Resample to target sample rate
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Trim silence from start and end
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        return y_trimmed, target_sr
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return np.zeros(target_sr), target_sr

def extract_f0_praat_like(y, sr=16000):
    """Extract F0 using autocorrelation method (Praat-like)"""
    try:
        # Use librosa's pyin for F0 estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) == 0:
            return 0.0, 0.0, 0.0
        
        mean_f0 = np.mean(f0_clean)
        std_f0 = np.std(f0_clean)
        median_f0 = np.median(f0_clean)
        
        return mean_f0, std_f0, median_f0
    except:
        return 0.0, 0.0, 0.0

def extract_jitter(y, sr=16000):
    """Extract jitter (period perturbation)"""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 2:
            return 0.0, 0.0
        
        periods = 1.0 / (f0_clean + 1e-8)
        
        if len(periods) < 2:
            return 0.0, 0.0
        
        # Local jitter (absolute)
        jitter_local = np.mean(np.abs(np.diff(periods))) / (np.mean(periods) + 1e-8)
        
        # Jitter RAP (Relative Average Perturbation)
        if len(periods) >= 3:
            rap = np.mean([abs(periods[i] - (periods[i-1] + periods[i+1])/2) 
                          for i in range(1, len(periods)-1)]) / (np.mean(periods) + 1e-8)
        else:
            rap = jitter_local
        
        return float(jitter_local), float(rap)
    except:
        return 0.0, 0.0

def extract_shimmer(y, sr=16000):
    """Extract shimmer (amplitude perturbation)"""
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Get amplitude envelope
        frame_length = int(sr * 0.025)  # 25ms frames
        hop_length_frame = int(sr * 0.010)  # 10ms hop
        amplitude = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length_frame)[0]
        
        if len(amplitude) < 2:
            return 0.0
        
        # Local shimmer
        shimmer_local = np.mean(np.abs(np.diff(amplitude))) / (np.mean(amplitude) + 1e-8)
        
        return shimmer_local
    except:
        return 0.0

def extract_hnr(y, sr=16000):
    """Extract Harmonics-to-Noise Ratio"""
    try:
        fft = np.fft.rfft(y)
        magnitude = np.abs(fft)
        power = magnitude ** 2
        
        # Find fundamental frequency
        f0_mean, _, _ = extract_f0_praat_like(y, sr)
        if f0_mean == 0:
            return 0.0
        
        # Estimate harmonics power vs noise power
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        harmonic_indices = []
        for h in range(1, 6):  # First 5 harmonics
            idx = np.argmin(np.abs(freqs - h * f0_mean))
            if idx < len(power):
                harmonic_indices.append(idx)
        
        harmonic_power = np.sum(power[harmonic_indices]) if harmonic_indices else 0
        total_power = np.sum(power)
        noise_power = total_power - harmonic_power
        
        if noise_power > 0:
            hnr_db = 10 * np.log10(harmonic_power / noise_power)
        else:
            hnr_db = 0.0
        
        return hnr_db
    except:
        return 0.0

def extract_voice_features(wav_path):
    """Extract all numeric voice features from audio file"""
    try:
        y, sr = load_and_preprocess_audio(wav_path)
        
        features = {}
        
        # F0 features
        mean_f0, std_f0, median_f0 = extract_f0_praat_like(y, sr)
        features['f0_mean'] = mean_f0
        features['f0_std'] = std_f0
        features['f0_median'] = median_f0
        
        # Jitter
        jitter_local, jitter_rap = extract_jitter(y, sr)
        features['jitter_local'] = jitter_local
        features['jitter_rap'] = jitter_rap
        
        # Shimmer
        shimmer_local = extract_shimmer(y, sr)
        features['shimmer_local'] = shimmer_local
        
        # HNR
        hnr = extract_hnr(y, sr)
        features['hnr'] = hnr
        
        # MFCCs (20 coefficients + delta + delta-delta)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfccs)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        
        # MFCC statistics
        for i in range(20):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
        
        # Energy
        energy = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = np.mean(energy)
        features['energy_std'] = np.std(energy)
        
        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    except Exception as e:
        print(f"Error extracting features from {wav_path}: {e}")
        raise

def predict_audio(audio_path, model_path):
    """
    Predict Parkinson's disease from audio recording
    
    Args:
        audio_path: Path to audio file
        model_path: Path to trained model
    
    Returns:
        label: Predicted class label
        parkinson_prob: Probability of Parkinson's disease
    """
    # Load model
    model = joblib.load(model_path)
    
    # Extract features from audio
    print(f"Processing audio: {audio_path}")
    features_dict = extract_voice_features(audio_path)
    
    # Select only the 15 most important features for Parkinson's detection
    # These are commonly used features based on research
    # ADJUST THIS LIST based on your actual training features!
    feature_order = [
        'f0_mean', 'f0_std', 'f0_median',      # Pitch features (3)
        'jitter_local', 'jitter_rap',           # Jitter (2)
        'shimmer_local',                        # Shimmer (1)
        'hnr',                                  # Harmonics-to-Noise (1)
        'mfcc_0_mean', 'mfcc_1_mean',          # First 2 MFCC means (2)
        'mfcc_0_std', 'mfcc_1_std',            # First 2 MFCC stds (2)
        'energy_mean', 'energy_std',            # Energy (2)
        'zcr_mean', 'zcr_std'                  # Zero crossing (2)
    ]
    # Total: 15 features
    
    # Create feature array in correct order
    try:
        features = np.array([features_dict[f] for f in feature_order])
        features = features.reshape(1, -1)
    except KeyError as e:
        print(f"Error: Missing feature {e}")
        print(f"Available features: {list(features_dict.keys())[:20]}")
        raise
    
    print(f"Feature vector shape: {features.shape}")
    print(f"Selected features: {feature_order}")
    print(f"Feature values: {features[0]}")
    
    # Prediction + probability
    pred = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    
    # Get Parkinson's probability (assuming 1 = parkinson, 0 = healthy)
    parkinson_prob = probs[1]
    label = "parkinson" if pred == 1 else "healthy"
    
    return label, parkinson_prob


def test_single_audio(audio_path, model_path):
    """Test a single audio recording"""
    
    try:
        label, prob = predict_audio(audio_path, model_path)
        
        print("\n" + "=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Audio Recording: {audio_path}")
        print(f"Prediction: {label.upper()}")
        print(f"Parkinson's likelihood: {prob*100:.2f}%")
        print(f"Healthy likelihood: {(1-prob)*100:.2f}%")
        print("-" * 60)
        
        if prob > 0.5:
            print("⚠️  Conclusion: Likely Parkinson's Disease")
        else:
            print("✓ Conclusion: Likely Healthy")
        print("=" * 60)
        
        return label, prob
        
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_multiple_audio(audio_paths, model_path):
    """Test multiple audio recordings and average results"""
    
    all_probs = []
    all_labels = []
    
    print("\n" + "=" * 60)
    print("TESTING MULTIPLE AUDIO RECORDINGS")
    print("=" * 60)
    
    for i, audio_path in enumerate(audio_paths, 1):
        print(f"\n[{i}/{len(audio_paths)}] Processing: {audio_path}")
        try:
            label, prob = predict_audio(audio_path, model_path)
            
            all_labels.append(label)
            all_probs.append(prob)
            
            print(f"  → Prediction: {label}")
            print(f"  → Parkinson's likelihood: {prob*100:.2f}%")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Calculate average probability
    if all_probs:
        avg_prob = np.mean(all_probs)
        
        print("\n" + "=" * 60)
        print("FINAL ASSESSMENT (AVERAGED ACROSS ALL RECORDINGS)")
        print("=" * 60)
        print(f"Number of recordings analyzed: {len(all_probs)}")
        print(f"Average Parkinson's likelihood: {avg_prob*100:.2f}%")
        print(f"Average Healthy likelihood: {(1-avg_prob)*100:.2f}%")
        print(f"Individual predictions: {all_labels}")
        print("-" * 60)
        
        if avg_prob > 0.5:
            print("⚠️  Conclusion: Likely Parkinson's Disease")
        else:
            print("✓ Conclusion: Likely Healthy")
        print("=" * 60)
        
        return all_labels, avg_prob
    else:
        print("\n✗ No valid predictions made")
        return None, None


def analyze_voice_features(audio_path):
    """Display extracted voice features for analysis"""
    
    print("\n" + "=" * 60)
    print("VOICE FEATURE ANALYSIS")
    print("=" * 60)
    
    try:
        y, sr = load_and_preprocess_audio(audio_path)
        
        print(f"Audio file: {audio_path}")
        print(f"Duration: {len(y)/sr:.2f} seconds")
        print(f"Sample rate: {sr} Hz")
        
        features = extract_voice_features(audio_path)
        
        print("\nVoice Quality Metrics:")
        print("-" * 60)
        print(f"F0 (Pitch) Mean:        {features['f0_mean']:.2f} Hz")
        print(f"F0 Std Dev:             {features['f0_std']:.2f} Hz")
        print(f"F0 Median:              {features['f0_median']:.2f} Hz")
        print(f"Jitter (Local):         {features['jitter_local']:.6f}")
        print(f"Jitter (RAP):           {features['jitter_rap']:.6f}")
        print(f"Shimmer (Local):        {features['shimmer_local']:.6f}")
        print(f"HNR:                    {features['hnr']:.2f} dB")
        print(f"Energy Mean:            {features['energy_mean']:.6f}")
        print(f"Zero Crossing Rate:     {features['zcr_mean']:.6f}")
        
        print("\nKey Indicators for Parkinson's:")
        print("-" * 60)
        
        # Typical indicators
        if features['jitter_local'] > 0.01:
            print("⚠️  High jitter detected (voice instability)")
        else:
            print("✓ Normal jitter levels")
            
        if features['shimmer_local'] > 0.05:
            print("⚠️  High shimmer detected (amplitude variation)")
        else:
            print("✓ Normal shimmer levels")
            
        if features['hnr'] < 10:
            print("⚠️  Low HNR (increased noise in voice)")
        else:
            print("✓ Normal HNR levels")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"Error analyzing features: {e}")


# Main execution
if __name__ == "__main__":
    
    import sys
    import os
    
    # Configuration
    MODEL_PATH = "models/random_forest_tuned.pkl"
    AUDIO_PATH = "sound/sample.wav"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Check if audio exists
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: Audio file not found at {AUDIO_PATH}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"\nPlease ensure your audio file is at: {AUDIO_PATH}")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("PARKINSON'S DISEASE AUDIO DETECTION TEST")
    print("=" * 60)
    
    # Test single audio recording
    test_single_audio(AUDIO_PATH, MODEL_PATH)
    
    # Show detailed voice features
    analyze_voice_features(AUDIO_PATH)
    
    print("\n\n")
    
    # Option: Test multiple recordings
    # Uncomment to test multiple audio files
    """
    print("Testing Multiple Audio Recordings")
    print("=" * 60)
    audio_files = [
        "sound/sample1.wav",
        "sound/sample2.wav",
        "sound/sample3.wav",
    ]
    test_multiple_audio(audio_files, MODEL_PATH)
    """