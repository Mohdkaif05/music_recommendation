<h1>Music Recommendation System </h1><br>
This project builds a music recommendation system that analyzes and compares songs based on their audio features. Using Librosa, it extracts features such as MFCCs, chroma, and spectral contrast from audio files. These features are stored in a dataset and used to recommend similar tracks. The system helps users discover new music based on sound similarity rather than just metadata.
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
