import React, { useRef, useState } from 'react';
import './VideoControl.css';

const VideoControl = () => {
    const videoRef = useRef(null);
    const streamRef = useRef(null);
    const mediaRecorderRef = useRef(null);
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState(null);
    const [detectedEmotion, setDetectedEmotion] = useState('');
    const [uploading, setUploading] = useState(false);

    const startRecording = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        streamRef.current = stream;
        if (videoRef.current) {
            videoRef.current.srcObject = stream;
        }

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
        };

        mediaRecorder.start();
        setIsRecording(true);
    };

    const stopRecording = () => {
        mediaRecorderRef.current.stop();
        streamRef.current.getTracks().forEach(track => track.stop());
        setIsRecording(false);
    };

    const uploadVideo = async () => {
        if (!recordedBlob) return;

        setUploading(true);

        const formData = new FormData();
        formData.append("video", recordedBlob, "recorded-video.webm");

        try {
            const response = await fetch('https://your-backend-url/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            setDetectedEmotion(data.detected_emotion || 'Unknown');
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
            videoRef.current.play();
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
                <button onClick={startRecording} disabled={isRecording} className="action-button start">
                    🎥 Start Recording
                </button>
                <button onClick={stopRecording} disabled={!isRecording} className="action-button stop">
                    🛑 Stop Recording
                </button>
            </div>

            <video ref={videoRef} autoPlay playsInline className={`video-preview ${isRecording || recordedBlob ? '' : 'hidden'}`} />

            {recordedBlob && (
                <div className="recorded-section">
                    <div className="button-container">
                        <button className="small-button" onClick={playRecordedVideo}>
                            ▶️ Play Your Message
                        </button>
                        <button className="small-button" onClick={uploadVideo} disabled={uploading}>
                            ☁️ Analyze My Message
                        </button>
                    </div>
                    {uploading && <p className="status">Analyzing your message...</p>}
                    {detectedEmotion && (
                        <p className="emotion">✅ Your Emotion: <strong>{detectedEmotion}</strong></p>
                    )}
                </div>
            )}

            <p className="footer">💬 We are here to help you. Follow the simple steps above.</p>
        </div>
    );
};

export default VideoControl;