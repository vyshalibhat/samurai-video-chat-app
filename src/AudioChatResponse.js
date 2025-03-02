import React, { useState, useEffect } from 'react';

const AudioChatResponse = ({ replyText }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [selectedVoice, setSelectedVoice] = useState(null);

    // Load voices and automatically select "Karen" (en-AU)
    useEffect(() => {
        const loadVoices = () => {
            const availableVoices = window.speechSynthesis.getVoices();

            const karenVoice = availableVoices.find(
                (voice) => voice.name.toLowerCase() === 'karen' && voice.lang === 'en-AU'
            );

            setSelectedVoice(karenVoice || availableVoices[0]);  // Fallback to first voice if Karen not found
        };

        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = loadVoices;
        }

        loadVoices();  // Initial check in case voices are already available
    }, []);

    const playReplyAudio = () => {
        if (!replyText) {
            alert('No reply available yet.');
            return;
        }

        const utterance = new SpeechSynthesisUtterance(replyText);

        if (selectedVoice) {
            utterance.voice = selectedVoice;
            console.log(`Using voice: ${selectedVoice.name} (${selectedVoice.lang})`);
        } else {
            console.warn("No preferred voice found. Using default system voice.");
        }

        utterance.onend = () => setIsPlaying(false);

        setIsPlaying(true);
        window.speechSynthesis.speak(utterance);
    };

    useEffect(() => {
        return () => window.speechSynthesis.cancel();  // Cleanup on unmount
    }, []);

    return (
        <div style={styles.container}>
            <h3>SamurAI's Response</h3>
            <button onClick={playReplyAudio} disabled={isPlaying} style={styles.button}>
                {isPlaying ? "Playing..." : "▶️ Play Reply"}
            </button>
        </div>
    );
};

const styles = {
    container: {
        marginTop: '20px',
        padding: '15px',
        borderRadius: '8px',
        backgroundColor: '#f9f9f9',
        border: '1px solid #ddd',
        textAlign: 'center',
    },
    button: {
        padding: '10px 20px',
        fontSize: '16px',
        cursor: 'pointer',
        backgroundColor: '#4CAF50',
        color: 'white',
        border: 'none',
        borderRadius: '5px',
        marginTop: '10px',
    },
};

export default AudioChatResponse;