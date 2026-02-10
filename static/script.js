document.addEventListener('DOMContentLoaded', () => {
    const gestureText = document.getElementById('gesture-text');
    const toggleSpeechBtn = document.getElementById('toggle-speech');
    
    let speechEnabled = false;
    let lastSpokenText = '';
    const synthesis = window.speechSynthesis;

    toggleSpeechBtn.addEventListener('click', () => {
        speechEnabled = !speechEnabled;
        toggleSpeechBtn.textContent = speechEnabled ? 'Disable Speech' : 'Enable Speech';
        toggleSpeechBtn.style.backgroundColor = speechEnabled ? '#FF5722' : '#4a90e2';
    });

    function fetchStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                gestureText.textContent = data.text;
                
                if (speechEnabled && data.text !== "Waiting..." && data.text !== "Unknown") {
                    speak(data.text);
                }
            })
            .catch(error => console.error('Error fetching status:', error));
    }

    function speak(text) {
        if (text !== lastSpokenText && !synthesis.speaking) {
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            utterance.pitch = 1.0;
            synthesis.speak(utterance);
            lastSpokenText = text;
            
            // Reset last spoken text after a delay to allow repeating same gesture
            setTimeout(() => {
                lastSpokenText = '';
            }, 3000);
        }
    }

    // Poll every 500ms
    setInterval(fetchStatus, 500);
});
