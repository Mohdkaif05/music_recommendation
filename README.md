<h1>Music Recommendation System </h1><br>
This project builds a music recommendation system that analyzes and compares songs based on their audio features. Using Librosa, it extracts features such as MFCCs, chroma, and spectral contrast from audio files. These features are stored in a dataset and used to recommend similar tracks. The system helps users discover new music based on sound similarity rather than just metadata.

<h2>Deployment</h2>

The API returns a playable URL for every recommendation. The current <code>index.json</code> stores audio files at:

<pre>https://satvat.pro/kaif-audio/songs/&lt;genre&gt;/&lt;filename&gt;</pre>

For example, <code>blues.00000.wav</code> is available at <code>https://satvat.pro/kaif-audio/songs/blues/blues.00000.wav</code>.

The web server must return <code>audio/wav</code> for WAV files and <code>audio/mpeg</code> for MP3 files. The current audio host returns <code>application/octet-stream</code>; configure its MIME types if the browser still cannot play the files.

Run the API from the project root so the relative <code>dataset/</code> and <code>models/</code> paths resolve correctly:

<pre>uvicorn main:app --host 0.0.0.0 --port 8000</pre>

Optional environment variables:

<ul>
<li><code>AUDIO_BASE_URL</code>: public audio folder URL.</li>
<li><code>AUDIO_INDEX_URL</code>: URL of the song metadata JSON. Defaults to <code>&lt;AUDIO_BASE_URL&gt;/index.json</code>.</li>
<li><code>AUDIO_INCLUDE_GENRE=true</code>: use <code>&lt;base-url&gt;/&lt;genre&gt;/&lt;filename&gt;</code> when audio is stored in genre subfolders.</li>
<li><code>CORS_ORIGINS=https://your-frontend.example</code>: comma-separated frontend origins. The default is <code>*</code> because the public API does not use cookies.</li>
</ul>

For local testing, the frontend uses <code>http://127.0.0.1:8000/predict</code>. For deployment, add this before loading <code>Gui/script.js</code>, replacing the value with the real FastAPI URL:

<pre>&lt;script&gt;window.API_URL = "https://your-api-domain.example/predict";&lt;/script&gt;
&lt;script src="script.js"&gt;&lt;/script&gt;</pre>

Do not use the frontend static-server URL for <code>API_URL</code>; that causes the “File does not reside within a trusted folder” 403 error.
<h2>Explanation of Each Feature</h2>

<h3>MFCC (Mel-Frequency Cepstral Coefficients)</h3>

Captures timbre (the texture or color of sound).<br>

Mimics how humans perceive pitch and tone.<br>

Useful for identifying instruments, singers, or genre.<br>

<h3>Chroma Features</h3>

Represents the 12 distinct pitch classes (C, C#, D, etc.).<br>

Helps understand harmonic and melodic content — good for recognizing chords and key.<br>

<h3>Spectral Contrast</h3>

Measures the difference between peaks and valleys in the sound spectrum.<br>

Captures how “bright” or “rich” a sound is — useful for genre and mood detection.<br>

<h3>Spectral Centroid</h3>

Indicates the center of mass of the spectrum, or where most of the energy is concentrated.<br>

Higher centroid → brighter sound; lower → darker sound.<br>

<h3>Tempo</h3>

Detects the speed or rhythm (beats per minute) of a track.<br>

Crucial for matching songs of similar energy or rhythm.<br>