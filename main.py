from fastapi import FastAPI, File, UploadFile, HTTPException  #type: ignore
from fastapi.middleware.cors import CORSMiddleware  #type: ignore
from pydantic import BaseModel, validator  #type: ignore
import numpy as np   #type: ignore
import pandas as pd  #type: ignore
import librosa   #type: ignore
import pickle
import os
import tempfile

app = FastAPI()
#---------------------cors middleware-----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # change ["*"] to your frontend URL for production (e.g., ["https://your-frontend.com"])
    allow_credentials=True,
    allow_methods=["POST"],  # allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # allow all headers
)

# -------------------- Load Dataset and Models --------------------
df = pd.read_csv("dataset/final_data.csv")  # original dataset with song info

# scaler + PCA
with open("models/scaler_pca.pkl", "rb") as f:
    scaler_pca = pickle.load(f)
scaler = scaler_pca["scaler"]
pca = scaler_pca["pca"]

# KMeans
with open("models/kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# KNN models per cluster
with open("models/knn.pkl", "rb") as f:
    knn = pickle.load(f)  # dict {cluster: KNN_model}


# -------------------- Pydantic Validation --------------------
class AudioValidator(BaseModel):
    filename: str
    content_type: str

    @validator("filename")
    def check_extension(cls, v):
        if not v.lower().endswith((".mp3", ".wav")):
            raise ValueError("Only .mp3 and .wav files are allowed.")
        return v

    @validator("content_type")
    def check_content_type(cls, v):
        if v not in ["audio/mpeg", "audio/wav"]:
            raise ValueError("Invalid file type.")
        return v


# -------------------- Feature Extraction --------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean, chroma_std = np.mean(chroma, axis=1), np.std(chroma, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean, contrast_std = np.mean(contrast, axis=1), np.std(contrast, axis=1)

    # Tonnetz
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    tonnetz_mean, tonnetz_std = np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)

    # Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_mean, centroid_std = np.mean(centroid), np.std(centroid)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

    # Combine all features
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        chroma_mean, chroma_std,
        contrast_mean, contrast_std,
        tonnetz_mean, tonnetz_std,
        [centroid_mean, centroid_std, zcr_mean, zcr_std]
    ])
    return feature_vector


# -------------------- Predict Endpoint --------------------
@app.post("/predict")
async def recommend_song(file: UploadFile = File(...)):
    # Step 1: Validate file
    try:
        validator = AudioValidator(filename=file.filename, content_type=file.content_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Step 2: Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Step 3: Extract features
        features = extract_features(tmp_path).reshape(1, -1)  # 2D array for sklearn

        # Step 4: Scale + PCA
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)

        # Step 5: Predict cluster
        cluster = int(kmeans.predict(features_pca))

        # Step 6: KNN recommendation
        if cluster not in knn:
            raise HTTPException(status_code=404, detail=f"No KNN model found for cluster {cluster}")

        model = knn[cluster]
        neighbors_idx = model.kneighbors(features_pca, n_neighbors=5, return_distance=False)

        # Step 7: Map neighbor indices to actual song names
        df_cluster = df[df['cluster'] == cluster].reset_index()
        recommended_songs = df_cluster.iloc[neighbors_idx[0]]['filename'].tolist()  # replace 'song_name' with your column

        return {
            "cluster": cluster,
            "recommended_songs": recommended_songs
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Step 8: Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

