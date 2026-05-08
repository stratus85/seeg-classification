import numpy as np
import mne
import hdf5storage
from sklearn.preprocessing import StandardScaler
from braindecode.datautil import create_from_mne_epochs
from scipy import signal
from scipy.integrate import trapezoid
from scipy.stats import skew, kurtosis
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
from preprocessing import dataset_to_numpy, save_or_preprocess
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import sys
import os
from datetime import datetime

# ============================================================================
# ALL YOUR EXISTING FEATURE EXTRACTION FUNCTIONS REMAIN EXACTLY THE SAME
# ============================================================================

def extract_frequency_features(X, fs=1000):
    """
    Extract frequency domain features from the paper
    X shape: (n_windows, n_channels, n_times)
    """
    n_windows, n_chans, n_times = X.shape
    
    # Features 13-17 from table = 5 features per channel
    n_features_per_chan = 5
    features = np.zeros((n_windows, n_chans * n_features_per_chan))
    
    for w in range(n_windows):
        for ch in range(n_chans):
            data = X[w, ch, :]
            
            # Compute power spectrum using Welch's method
            freqs, psd = signal.welch(data, fs=fs, nperseg=min(256, n_times))
            
            # Feature 13: Sub-band power ratio (60-140 Hz / 0-60 Hz)
            low_band = (freqs >= 0) & (freqs <= 60)
            high_band = (freqs >= 60) & (freqs <= 140)
            
            low_power = trapezoid(psd[low_band], freqs[low_band])
            high_power = trapezoid(psd[high_band], freqs[high_band])
            
            # Avoid division by zero
            spr = high_power / low_power if low_power > 0 else 0

            # Features 14-15: PSD and ASD (we already have PSD from welch)
            # ASD is sqrt(PSD) * sqrt(T) approximation
            asd = np.sqrt(psd * n_times)
            
            # Feature 16: Spectral Centroid
            total_power = np.sum(psd)
            if total_power > 0:
                sc = np.sum(freqs * psd) / total_power
            else:
                sc = 0
            
            # Feature 17: Spectral Kurtosis
            if total_power > 0:
                # Calculate 4th moment
                fourth_moment = np.sum(((freqs - sc) ** 4) * psd)
                # Normalize by standard deviation^4
                sp = np.sqrt(np.sum(((freqs - sc) ** 2) * psd) / total_power)
                sk = fourth_moment / (sp**4 * total_power) if sp > 0 else 0
            else:
                sk = 0
            
            # Store features for this channel
            base_idx = ch * n_features_per_chan
            features[w, base_idx] = spr           # Feature 13
            features[w, base_idx + 1] = np.mean(psd)  # Feature 14 (mean PSD)
            features[w, base_idx + 2] = np.mean(asd)  # Feature 15 (mean ASD)
            features[w, base_idx + 3] = sc        # Feature 16
            features[w, base_idx + 4] = sk        # Feature 17
    
    return features


def extract_band_powers(X, fs=1000):
    """Extract power in standard frequency bands"""
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma1': (30, 50),
        'gamma2': (50, 70),
        'gamma3': (70, 90),
        'gamma4': (90, 110),
        'high_gamma1': (110, 140),
        'high_gamma2': (140, 170),
        'high_gamma3': (170, 200)
    }
    
    n_windows, n_chans, n_times = X.shape
    n_bands = len(bands)
    
    features = np.zeros((n_windows, n_chans * n_bands))
    
    for w in range(n_windows):
        for ch in range(n_chans):
            freqs, psd = signal.welch(X[w, ch, :], fs=fs, nperseg=min(256, n_times))
            
            for b_idx, (band_name, (low, high)) in enumerate(bands.items()):
                band_mask = (freqs >= low) & (freqs <= high)
                band_power = trapezoid(psd[band_mask], freqs[band_mask])
                features[w, ch * n_bands + b_idx] = band_power
    
    return features


def extract_time_domain_features(X):
    """
    Extract comprehensive time-domain features for EEG/SEEG
    X shape: (n_windows, n_channels, n_times)
    Returns: (n_windows, n_channels * 17) features
    """
    n_windows, n_chans, n_times = X.shape
    
    # 17 time-domain features per channel
    n_features_per_chan = 17
    features = np.zeros((n_windows, n_chans * n_features_per_chan))
    
    for w in range(n_windows):
        for ch in range(n_chans):
            data = X[w, ch, :]
            base_idx = ch * n_features_per_chan
            
            # ===== BASIC STATISTICAL FEATURES =====
            features[w, base_idx] = np.mean(data)           # 0: Mean
            features[w, base_idx + 1] = np.var(data)        # 1: Variance
            features[w, base_idx + 2] = np.std(data)        # 2: Std
            features[w, base_idx + 3] = np.sqrt(np.mean(data**2))  # 3: RMS
            features[w, base_idx + 4] = np.ptp(data)        # 4: Peak-to-peak
            
            # ===== HIGHER-ORDER MOMENTS =====
            features[w, base_idx + 5] = skew(data)          # 5: Skewness
            features[w, base_idx + 6] = kurtosis(data)      # 6: Kurtosis
            
            # ===== SIGNAL COMPLEXITY FEATURES =====
            # Zero-crossing rate
            zero_crossings = np.where(np.diff(np.signbit(data)))[0]
            features[w, base_idx + 7] = len(zero_crossings) / n_times  # 7: ZCR
            
            # Line length
            features[w, base_idx + 8] = np.sum(np.abs(np.diff(data)))  # 8: Line length
            
            # Teager-Kaiser energy (average)
            tkeo = 0
            for i in range(1, n_times - 1):
                tkeo += data[i]**2 - data[i+1] * data[i-1]
            features[w, base_idx + 9] = tkeo / (n_times - 2)  # 9: Avg TKEO
            
            # ===== HJORTH PARAMETERS =====
            diff_data = np.diff(data)
            var_data = np.var(data)
            var_diff = np.var(diff_data)
            
            # Mobility
            if var_data > 0:
                mobility = np.sqrt(var_diff / var_data)
            else:
                mobility = 0
            features[w, base_idx + 10] = mobility  # 10: Mobility
            
            # Complexity
            diff2_data = np.diff(diff_data)
            var_diff2 = np.var(diff2_data)
            if var_diff > 0 and mobility > 0:
                mobility_diff = np.sqrt(var_diff2 / var_diff)
                complexity = mobility_diff / mobility
            else:
                complexity = 0
            features[w, base_idx + 11] = complexity  # 11: Complexity
            
            # ===== PERCENTILE FEATURES =====
            features[w, base_idx + 12] = np.percentile(data, 25)  # 12: Q1
            features[w, base_idx + 13] = np.percentile(data, 50)  # 13: Median (Q2)
            features[w, base_idx + 14] = np.percentile(data, 75)  # 14: Q3
            features[w, base_idx + 15] = features[w, base_idx + 14] - features[w, base_idx + 12]  # 15: IQR
            
            # ===== MIN/MAX FEATURES =====
            features[w, base_idx + 16] = np.min(data)  # 16: Min
    
    return features


def extract_band_limited_time_features(X, fs=1000):
    """
    Extract time-domain features from band-limited signals
    
    Based on research showing that Hjorth parameters and statistical moments
    in specific frequency bands improve classification
    
    X shape: (n_windows, n_channels, n_times)
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 70),
        'high_gamma': (70, 140)
    }
    
    n_windows, n_chans, n_times = X.shape
    n_bands = len(bands)
    n_features_per_band = 3  # RMS, mobility, complexity
    
    features = np.zeros((n_windows, n_chans * n_bands * n_features_per_band))
    
    from scipy.fft import fft, ifft, fftfreq
    
    for w in range(n_windows):
        for ch in range(n_chans):
            data = X[w, ch, :]
            n = len(data)
            
            # FFT for efficient bandpass filtering
            fft_data = fft(data)
            fft_freqs = fftfreq(n, 1/fs)
            
            for b_idx, (band_name, (low, high)) in enumerate(bands.items()):
                # Bandpass filter in frequency domain
                band_mask = (np.abs(fft_freqs) >= low) & (np.abs(fft_freqs) <= high)
                fft_filtered = fft_data * band_mask
                band_signal = np.real(ifft(fft_filtered))
                
                base_idx = (ch * n_bands + b_idx) * n_features_per_band
                
                # RMS in this band
                features[w, base_idx] = np.sqrt(np.mean(band_signal**2))
                
                # Hjorth mobility in this band
                diff_band = np.diff(band_signal)
                if np.var(band_signal) > 0:
                    mobility = np.sqrt(np.var(diff_band) / np.var(band_signal))
                else:
                    mobility = 0
                features[w, base_idx + 1] = mobility
                
                # Hjorth complexity in this band
                diff2_band = np.diff(diff_band)
                if np.var(diff_band) > 0 and mobility > 0:
                    mobility_diff = np.sqrt(np.var(diff2_band) / np.var(diff_band))
                    complexity = mobility_diff / mobility
                else:
                    complexity = 0
                features[w, base_idx + 2] = complexity
    
    return features


class Tee:
    """Class to redirect stdout to both file and console"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


def run_svm_for_pn(pn, results_dir):
    """
    Run SVM training for a specific pn value and save output to file.
    Returns True if successful, False if error occurred.
    """
    # Create output file path
    output_file = os.path.join(results_dir, f'pn_{pn}_output.txt')
    
    # Redirect stdout to both console and file
    tee = Tee(output_file)
    original_stdout = sys.stdout
    sys.stdout = tee
    
    try:
        print("=" * 80)
        print(f"Starting SVM run for pn = {pn}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # Preprocess data using your imported preprocess function
        train_set, valid_set, test_set = save_or_preprocess(pn, cache_dir='../Preprocessed')
        
        # Convert datasets to numpy
        X_train, y_train = dataset_to_numpy(train_set)
        X_val, y_val = dataset_to_numpy(valid_set)
        X_test, y_test = dataset_to_numpy(test_set)
        
        print(f"\nData shapes:")
        print(f"  Train windows: {X_train.shape}")
        print(f"  Validation windows: {X_val.shape}")
        print(f"  Test windows: {X_test.shape}")
        
        # Combine training and validation data
        X_train_full = np.vstack([X_train, X_val])
        y_train_full = np.concatenate([y_train, y_val])
        
        print(f"\nCombined training data shape: {X_train_full.shape}")
        
        # ============================================================================
        # EXTRACT ALL FEATURE SETS
        # ============================================================================
        
        print("\n" + "="*60)
        print("EXTRACTING FEATURES")
        print("="*60)
        
        # 1. Frequency domain features
        print("\n1. Extracting frequency features...")
        X_train_freq = extract_frequency_features(X_train_full)
        X_test_freq = extract_frequency_features(X_test)
        print(f"   Frequency features shape: {X_train_freq.shape}")
        
        # 2. Band powers
        print("2. Extracting band powers...")
        X_train_bands = extract_band_powers(X_train_full)
        X_test_bands = extract_band_powers(X_test)
        print(f"   Band powers shape: {X_train_bands.shape}")
        
        # 3. Time-domain features
        print("3. Extracting time-domain features...")
        X_train_time = extract_time_domain_features(X_train_full)
        X_test_time = extract_time_domain_features(X_test)
        print(f"   Time-domain features shape: {X_train_time.shape}")
        
        # 4. Band-limited time features
        print("4. Extracting band-limited time features...")
        X_train_bt = extract_band_limited_time_features(X_train_full)
        X_test_bt = extract_band_limited_time_features(X_test)
        print(f"   Band-limited time features shape: {X_train_bt.shape}")
        
        # Combine ALL features
        print("\nCombining all features...")
        X_train_feat = np.hstack([X_train_freq, X_train_bands, X_train_time, X_train_bt])
        X_test_feat = np.hstack([X_test_freq, X_test_bands, X_test_time, X_test_bt])
        
        print(f"\nTotal feature dimensions:")
        print(f"  Frequency: {X_train_freq.shape[1]}")
        print(f"  Band powers: {X_train_bands.shape[1]}")
        print(f"  Time-domain: {X_train_time.shape[1]}")
        print(f"  Band-limited time: {X_train_bt.shape[1]}")
        print(f"  TOTAL: {X_train_feat.shape[1]} features per window")
        
        # ============================================================================
        # TUNE HYPERPARAMETERS WITH GRIDSEARCHCV
        # ============================================================================
        
        print("\n" + "="*60)
        print("TUNING HYPERPARAMETERS WITH GRIDSEARCHCV")
        print("="*60)
        
        # Create pipeline with feature selection and SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('select', SelectKBest(mutual_info_classif)),
            ('svm', LinearSVC(
                class_weight='balanced',
                random_state=42,
                max_iter=20000,
                dual='auto'
            ))
        ])
        
        # Define parameter grid
        param_grid = {
            'select__k': [500, 750, 1000],  # Tune number of features
            'svm__C': [0.01, 0.1, 1.0],  # Tune regularization
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,  # 5-fold cross-validation on training data
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,  # Set to 0 for clean output
            return_train_score=True
        )
        
        # Fit on combined training data
        grid_search.fit(X_train_feat, y_train_full)
        
        # Print best results
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f} (+/- {grid_search.cv_results_['std_test_score'][grid_search.best_index_]*2:.4f})")
        
        # Get the best model
        best_model = grid_search.best_estimator_
        
        # Evaluate on test set
        test_accuracy = best_model.score(X_test_feat, y_test)
        print(f"\nTest accuracy with best model: {test_accuracy:.4f}")
        
        # Detailed test evaluation
        y_pred = best_model.predict(X_test_feat)
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Gesture0', 'Gesture1', 'Gesture2', 'Gesture3', 'Gesture4']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Optional: Show feature selection results from best model
        selector = best_model.named_steps['select']
        selected_indices = selector.get_support(indices=True)
        
        # Calculate feature counts
        freq_dim = X_train_freq.shape[1]
        bands_dim = freq_dim + X_train_bands.shape[1]
        time_dim = bands_dim + X_train_time.shape[1]
        
        freq_selected = sum(1 for i in selected_indices if i < freq_dim)
        bands_selected = sum(1 for i in selected_indices if freq_dim <= i < bands_dim)
        time_selected = sum(1 for i in selected_indices if bands_dim <= i < time_dim)
        bt_selected = sum(1 for i in selected_indices if i >= time_dim)
        
        print("\n" + "="*60)
        print("FEATURE SELECTION RESULTS")
        print("="*60)
        print(f"Selected features - Frequency: {freq_selected}, Bands: {bands_selected}, "
              f"Time: {time_selected}, BT: {bt_selected}")
        print(f"Total features selected: {len(selected_indices)}")
        
        print("\n" + "="*80)
        print(f"Finished SVM run for pn = {pn}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        return True
        
    except Exception as e:
        print("=" * 80)
        print(f"ERROR for pn = {pn}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        print("=" * 80)
        return False
        
    finally:
        # Restore original stdout and close the tee
        sys.stdout = original_stdout
        tee.close()


# ============================================================================
# MAIN EXECUTION - BATCH PROCESSING
# ============================================================================

if __name__ == '__main__':
    # Define results directory
    results_dir = '../SVM results/reruns'
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Define pn values to run (2 to 41)
    # Skip any that are known to not work - you can add to this list as needed
    skip_pn = []  # Add problematic pn values here, e.g., [12, 15, 23]
    pn_values = [41]
    
    print("=" * 80)
    print(f"Starting batch processing for SVM on pn values: {pn_values}")
    print(f"Results will be saved to: {results_dir}")
    print(f"Total runs: {len(pn_values)}")
    print("=" * 80)
    
    # Track results
    successful = []
    failed = []
    
    # Run for each pn value
    for pn in pn_values:
        print(f"\n{'=' * 80}")
        print(f"Processing pn = {pn}...")
        print(f"{'=' * 80}")
        
        success = run_svm_for_pn(pn, results_dir)
        
        if success:
            successful.append(pn)
        else:
            failed.append(pn)
        
        print(f"\nCompleted pn = {pn} (Success: {success})")
    
    # Print summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {successful}")
    print(f"Failed runs: {failed}")
    print(f"Total successful: {len(successful)}/{len(pn_values)}")
    print(f"Results saved in: {results_dir}")
    print("=" * 80)