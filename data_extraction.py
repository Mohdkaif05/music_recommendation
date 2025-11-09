import librosa  # type: ignore
import numpy as np
import pandas as pd # type: ignore
import os
from glob import glob
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # type: ignore


def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)
        
        # Optional: remove silence
        y, _ = librosa.effects.trim(y)

        # === 1️⃣ MFCCs ===
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)

        # === 2️⃣ Chroma features ===
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # === 3️⃣ Spectral Contrast ===
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        # === 4️⃣ Tonnetz ===
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        tonnetz_mean = np.mean(tonnetz, axis=1)
        tonnetz_std = np.std(tonnetz, axis=1)

        # === 5️⃣ Spectral Centroid ===
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std = np.std(centroid)

        # === 6️⃣ Zero Crossing Rate ===
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        # === Combine all features ===
        feature_vector = np.hstack([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            tonnetz_mean, tonnetz_std,
            [centroid_mean, centroid_std, zcr_mean, zcr_std]
        ])

        return feature_vector

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(13*2 + 12*2 + 7*2 + 6*2 + 4)  # placeholder with correct length


if __name__ == "__main__":
    path = r"D:\recomdn_engine\Data\genres_original\*\*.wav"
    files = glob(path)

    # Multiprocessing setup
    num_cores = cpu_count() - 2  # keep 2 cores free
    print(f"Using {num_cores} CPU cores for parallel processing...")

    # Run multiprocessing with progress bar
    with Pool(num_cores) as p:
        results = list(tqdm(p.imap(extract_features, files), total=len(files)))

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df['filename'] = [os.path.basename(f) for f in files]
    df['genre'] = [os.path.basename(os.path.dirname(f)) for f in files]



    # Save to CSV
    df.to_csv("audio_features_updated.csv", index=False)
    print("Feature extraction complete! Saved to 'audio_features.csv'")


