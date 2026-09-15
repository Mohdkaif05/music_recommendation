from fastapi import FastAPI, File, UploadFile, HTTPException  #type: ignore
from fastapi.middleware.cors import CORSMiddleware  #type: ignore
from pydantic import BaseModel, validator  #type: ignore
import numpy as np   #type: ignore
import pandas as pd  #type: ignore
import librosa   #type: ignore
import pickle
import os
import tempfile
import json
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.getenv("PORT", "8000"))

app = FastAPI(title="Music Recommendation API")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

audio_base_url = os.getenv("AUDIO_BASE_URL", "https://satvat.pro/kaif-audio/songs").rstrip("/")
audio_index_url = os.getenv("AUDIO_INDEX_URL", f"{audio_base_url}/index.json")
audio_include_genre = os.getenv("AUDIO_INCLUDE_GENRE", "false").lower() == "true"
audio_metadata_cache = None
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]


def build_audio_url(genre, filename):
    """Build a public URL matching the hosted audio folder structure."""
    genre_path = f"/{quote(str(genre), safe='')}" if audio_include_genre else ""
    return f"{audio_base_url}{genre_path}/{quote(str(filename), safe='')}"


def load_audio_metadata():
    """Load index.json as a filename-to-metadata map without blocking recommendations on it."""
    global audio_metadata_cache
    if audio_metadata_cache is not None:
        return audio_metadata_cache

    audio_metadata_cache = {}
    try:
        with urlopen(audio_index_url, timeout=5) as response:
            payload = json.load(response)

        if isinstance(payload, dict):
            entries = payload.get("songs", payload.get("files", payload))
            if isinstance(entries, dict):
                entries = [dict(value, filename=key) if isinstance(value, dict) else {"filename": key, "value": value}
                           for key, value in entries.items()]
        else:
            entries = payload

        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            filename = (
                entry.get("filename")
                or entry.get("file")
                or entry.get("path")
                or entry.get("name")
            )
            relative_url = entry.get("url") or entry.get("audio_url")
            keys = []
            if filename:
                keys.append(os.path.basename(str(filename)))
            if relative_url:
                keys.append(os.path.basename(str(relative_url)))
            for key in keys:
                audio_metadata_cache[key] = entry
    except Exception:
        pass

    return audio_metadata_cache


#---------------------cors middleware-----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],  # allow all headers
)

# -------------------- Load Dataset and Models --------------------
df = pd.read_csv(BASE_DIR / "dataset" / "final_data.csv")  # original dataset with song info

# scaler + PCA
with open(BASE_DIR / "models" / "scaler_pca.pkl", "rb") as f:
    scaler_pca = pickle.load(f)
scaler = scaler_pca["scaler"]
pca = scaler_pca["pca"]

# KMeans
with open(BASE_DIR / "models" / "kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

# KNN models per cluster
with open(BASE_DIR / "models" / "knn.pkl", "rb") as f:
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
        cluster_pred = np.asarray(kmeans.predict(features_pca)).ravel()
        if cluster_pred.size == 0:
            raise HTTPException(status_code=500, detail="Prediction returned no cluster label.")
        cluster = int(cluster_pred[0])

        # Step 6: KNN recommendation
        if cluster not in knn:
            raise HTTPException(status_code=404, detail=f"No KNN model found for cluster {cluster}")

        model = knn[cluster]
        # Ask for one extra result so the uploaded song can be excluded.
        df_cluster = df[df['cluster'] == cluster].reset_index()
        neighbor_count = min(6, len(df_cluster))
        neighbors_idx = model.kneighbors(features_pca, n_neighbors=neighbor_count, return_distance=False)
        uploaded_filename = os.path.basename(file.filename).lower()
        recommended_indices = [
            index for index in neighbors_idx[0]
            if str(df_cluster.iloc[index]['filename']).lower() != uploaded_filename
        ][:5]
        recommended_rows = df_cluster.iloc[recommended_indices][['filename', 'genre']]
        print(recommended_rows)
        metadata_map = load_audio_metadata()
        recommended_songs = [
            {
                "filename": row.filename,
                "genre": row.genre,
                # "url": metadata_audio_url(
                #     metadata_map.get(str(row.filename), {}), row.genre, row.filename
                # )
                "url": f"{audio_base_url}/{row.genre}/{row.filename}",
                "metadata": metadata_map.get(str(row.filename), {}),
            }
            for row in recommended_rows.itertuples(index=False)
        ]

        return {
            "cluster": cluster,
            "recommended_songs": recommended_songs
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    finally:
        # Step 8: Cleanup temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT)
