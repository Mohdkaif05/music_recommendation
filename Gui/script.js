document.addEventListener('DOMContentLoaded', () => {
    const audioInput = document.getElementById('audioFile');
    const recommendBtn = document.getElementById('recommendBtn');
    const list = document.getElementById('recommendationList');
    const uploadedAudioPlayer = document.getElementById('uploadedAudioPlayer');
    const uploadedSongContainer = document.getElementById('uploadedSongContainer');
    const uploadedSongName = document.getElementById('uploadedSongName');

    const apiurl = window.API_URL || 'http://127.0.0.1:8000/predict';
    const audioBaseUrl = 'https://satvat.pro/kaif-audio/';

    function metadataLabel(metadata) {
        const fields = [
            ['title', 'Title'],
            ['artist', 'Artist'],
            ['album', 'Album'],
            ['genre', 'Genre'],
            ['year', 'Year']
        ];
        return fields
            .filter(([key]) => metadata && metadata[key] !== undefined && metadata[key] !== '')
            .map(([key, label]) => `${label}: ${metadata[key]}`)
            .join(' | ');
    }


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
            });

            if (!response.ok) {
                const responseBody = await response.text();
                let detail = responseBody;
                try {
                    const errorData = JSON.parse(responseBody);
                    detail = errorData.detail ? `: ${errorData.detail}` : '';
                } catch {
                    // Keep the plain-text response when the server did not return JSON.
                }
                throw new Error(`Server error: ${response.status}${detail ? ` - ${detail}` : ''}`);
            }

            const data = await response.json();
            console.log("✅ Recommended songs:", data);

            list.innerHTML = ''; // Clear loading
            data.recommended_songs.forEach((song, index) => {
                const li = document.createElement('li');
                const label = document.createElement('span');
                const metadata = metadataLabel(song.metadata);
                label.textContent = `${index + 1}. ${song.filename}${metadata ? ` (${metadata})` : ''}`;

                const player = document.createElement('audio');
                console.log(song)
                player.controls = true;
                player.preload = 'none';
                const metadataUrl = song.metadata && song.metadata.url;
                const relativeSongUrl = String(metadataUrl || song.url || '').replace(/^\/+/, '');
                player.src = /^https?:\/\//i.test(relativeSongUrl)
                    ? relativeSongUrl
                    : `${audioBaseUrl}${relativeSongUrl}`;
                const extension = song.filename.toLowerCase().split('.').pop();
                player.setAttribute('type', extension === 'mp3' ? 'audio/mpeg' : 'audio/wav');
                player.setAttribute('aria-label', `Play ${song.filename}`);
                player.addEventListener('error', () => {
                    label.textContent = `${index + 1}. ${song.filename} (audio unavailable)`;
                });

                li.append(label, player);
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