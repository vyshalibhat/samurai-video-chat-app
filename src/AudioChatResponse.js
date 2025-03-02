
import React, { useState, useEffect } from 'react';
import './AudioChatResponse.css';

const AudioChatResponse = ({ replyText }) => {
    const [isPlaying, setIsPlaying] = useState(false);
    const [showText, setShowText] = useState(false);

    const playReplyAudio = () => {
        if (!replyText) return;
        
        setIsPlaying(true);
        
        const utterance = new SpeechSynthesisUtterance(replyText);
        utterance.rate = 0.9; // Slightly slower rate for better understanding
        utterance.pitch = 1.1; // Slightly higher pitch for a friendly tone
        utterance.volume = 1.0;
        
        // Try to use a more natural voice if available
        const voices = window.speechSynthesis.getVoices();
        const preferredVoices = voices.filter(voice => 
            voice.name.includes('Samantha') || 
            voice.name.includes('Google') || 
            voice.name.includes('Natural')
        );
        
        if (preferredVoices.length > 0) {
            utterance.voice = preferredVoices[0];
        }
        
        utterance.onend = () => setIsPlaying(false);
        utterance.onerror = () => {
            setIsPlaying(false);
            alert('Sorry, there was an error playing the audio. Please try again.');
        };
        
        window.speechSynthesis.speak(utterance);
    };
    
    // Toggle showing the text
    const toggleShowText = () => {
        setShowText(!showText);
    };

    useEffect(() => {
        return () => window.speechSynthesis.cancel();  // Cleanup on unmount
    }, []);

    return (
        <div className="chat-response-container">
            <h3>SamurAI's Response</h3>
            <button 
                onClick={playReplyAudio} 
                disabled={isPlaying} 
                className="play-button"
            >
                <span className="play-button-icon">{isPlaying ? "🔊" : "▶️"}</span>
                {isPlaying ? "Playing..." : "Play Reply"}
            </button>
            
            <button onClick={toggleShowText} className="show-text-button">
                {showText ? "Hide Text" : "Show Text"}
            </button>
            
            {showText && (
                <div className="response-text" style={{ display: showText ? 'block' : 'none' }}>
                    {replyText}
                </div>
            )}
        </div>
    );
};

export default AudioChatResponse;
