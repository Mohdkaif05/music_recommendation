document.addEventListener('DOMContentLoaded', () => {
    const audioInput = document.getElementById('audioFile');
    const recommendBtn = document.getElementById('recommendBtn');
    const list = document.getElementById('recommendationList');
    const uploadedAudioPlayer = document.getElementById('uploadedAudioPlayer');
    const uploadedSongContainer = document.getElementById('uploadedSongContainer');
    const uploadedSongName = document.getElementById('uploadedSongName');
    const apiurl = "http://127.0.0.1:8000/predict";

    // 1. Handle uploaded song playback
    audioInput.addEventListener('change', () => {
        const file = audioInput.files[0];
        if (file && (file.name.endsWith('.mp3') || file.name.endsWith('.wav'))) {
            const fileURL = URL.createObjectURL(file);
            uploadedAudioPlayer.src = fileURL;
            uploadedSongName.textContent = file.name;
            uploadedSongContainer.style.display = 'block';
            list.innerHTML = `<li class="placeholder">Ready to analyze: ${file.name}. Click 'Get Recommendations'.</li>`;
        } else {
            uploadedSongContainer.style.display = 'none';
            uploadedAudioPlayer.src = '';
            alert("Invalid file type. Please select an MP3 or WAV file.");
        }
    });

    // 2. Handle recommendation request
    recommendBtn.addEventListener('click', async () => {
        const file = audioInput.files[0];

        if (!file) {
            alert("Please select an audio file first.");
            return;
        }

        recommendBtn.disabled = true;
        list.innerHTML = `<li class="placeholder">Processing song: ${file.name}...</li>`;

        // ✅ Use FormData for file upload
        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch(apiurl, {
                method: 'POST',
                body: formData,
                credentials: "include", // ✅ Send cookies / credentials with request
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            console.log("✅ Recommended songs:", data);

            list.innerHTML = ''; // Clear loading
            data.recommended_songs.forEach((song, index) => {
                const li = document.createElement('li');
                li.textContent = `${index + 1}. ${song}`;
                list.appendChild(li);
            });
        } catch (error) {
            console.error("❌ Error occurred:", error.message);
            list.innerHTML = `<li class="placeholder error">Error: ${error.message}</li>`;
        } finally {
            recommendBtn.disabled = false;
        }
    });
});
