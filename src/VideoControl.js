
import React, { useRef, useState } from 'react';
import './VideoControl.css';

const VideoControl = () => {
    const videoRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [recordedBlob, setRecordedBlob] = useState(null);
    const [isRecording, setIsRecording] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [detectedEmotion, setDetectedEmotion] = useState('');

    const startRecording = async () => {
        try {
            console.log("Requesting webcam access...");
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
            console.log("Webcam stream obtained:", stream);
            
            videoRef.current.srcObject = stream;
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            
            const chunks = [];
            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    chunks.push(event.data);
                }
            };
            
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, { type: 'video/webm' });
                setRecordedBlob(blob);
                
                // Stop all tracks of the stream
                stream.getTracks().forEach(track => track.stop());
                videoRef.current.srcObject = null;
            };
            
            mediaRecorder.start();
            setIsRecording(true);
        } catch (error) {
            console.error("Error accessing media devices:", error);
            alert("Could not access camera or microphone. Please ensure you've granted permission and try again.");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
        }
    };

    const uploadVideo = async () => {
        if (!recordedBlob) {
            alert('No recorded video available.');
            return;
        }

        setUploading(true);
        const formData = new FormData();
        formData.append("video", recordedBlob, "recorded-video.webm");

        try {
            // For testing, we'll just simulate a successful response
            // If you have a real backend, replace this URL with your actual endpoint
            console.log("Video would be uploaded to backend");
            
            // Simulating a response
            const fakeEmotions = ["Happy", "Neutral", "Concerned", "Thoughtful"];
            const randomEmotion = fakeEmotions[Math.floor(Math.random() * fakeEmotions.length)];
            
            setDetectedEmotion(randomEmotion);
            
            /* Uncomment when you have a real backend
            const response = await fetch('https://your-backend-url/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            setDetectedEmotion(data.detected_emotion || 'Unknown');
            */
        } catch (error) {
            console.error("Upload failed", error);
            alert("We could not process your message. Please try again.");
        } finally {
            setUploading(false);
        }
    };

    const playRecordedVideo = () => {
        if (recordedBlob && videoRef.current) {
            const videoURL = URL.createObjectURL(recordedBlob);
            videoRef.current.srcObject = null;
            videoRef.current.src = videoURL;
            videoRef.current.controls = true;
            
            // Use a promise and catch any errors that might occur
            videoRef.current.play()
                .catch(error => {
                    console.error("Error playing video:", error);
                    alert("Could not play the video. Please try recording again.");
                });
        } else {
            alert('No recorded video available.');
        }
    };

    return (
        <div className="dementia-container">
            <h1>Welcome to SamurAI</h1>
            <p className="instructions">👋 Let's record your message so we can understand how you're feeling today.</p>
            <p className="instructions">Press <strong>"Start Recording"</strong>, talk to us, then press <strong>"Stop Recording"</strong>.</p>

            <div className="button-container">
                <button 
                    onClick={startRecording} 
                    disabled={isRecording} 
                    className={`action-button start ${isRecording ? 'pulse' : ''}`}
                >
                    {isRecording ? '📹 Recording...' : '🎥 Start Recording'}
                </button>
                <button 
                    onClick={stopRecording} 
                    disabled={!isRecording} 
                    className="action-button stop"
                >
                    🛑 Stop Recording
                </button>
            </div>

            <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                className={`video-preview ${isRecording || recordedBlob ? '' : 'hidden'}`} 
            />
            
            {recordedBlob && !isRecording && (
                <div className="recorded-section">
                    <button onClick={playRecordedVideo} className="small-button">
                        ▶️ Play Recorded Video
                    </button>
                    <button onClick={uploadVideo} disabled={uploading} className="small-button">
                        {uploading ? '⏳ Processing...' : '✅ Analyze My Mood'}
                    </button>
                </div>
            )}
            
            {detectedEmotion && (
                <div className="emotion">
                    Detected Mood: <span className="highlighted-emotion">{detectedEmotion}</span>
                </div>
            )}

            <p className="footer">💬 We are here to help you. Follow the simple steps above.</p>
        </div>
    );
};

export default VideoControl;
