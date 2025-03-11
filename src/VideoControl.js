// VideoControl.js
import React, { useRef, useState } from "react";
import "./VideoControl.css";

const VideoControl = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  // Recorded data
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);

  // Outputs
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const [transcribedText, setTranscribedText] = useState("");
  const [llmResponse, setLlmResponse] = useState("");

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;

      const mimeTypes = [
        "video/webm;codecs=vp8",
        "video/webm;codecs=vp9",
        "video/mp4",
        "video/avi",
      ];

      let chosenType = "";
      for (const type of mimeTypes) {
        if (MediaRecorder.isTypeSupported(type)) {
          chosenType = type;
          break;
        }
      }

      let mediaRecorder;
      if (!chosenType) {
        console.warn("No specified MIME types are supported, using default");
        mediaRecorder = new MediaRecorder(stream);
      } else {
        console.log("Using MIME type:", chosenType);
        mediaRecorder = new MediaRecorder(stream, { mimeType: chosenType });
      }
      mediaRecorderRef.current = mediaRecorder;

      const chunks = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType });
        setRecordedBlob(blob);
        // Stop camera
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // ------------------------------------------------------------------
  // 1) Upload for Emotion => POST /predict
  //    We only reset "detectedEmotion" so old transcription & LLM remain
  // ------------------------------------------------------------------
  const handleUploadForEmotion = async () => {
    if (!recordedBlob) {
      alert("No recorded video available.");
      return;
    }
    setDetectedEmotion("(loading...)");

    const formData = new FormData();
    formData.append("file", recordedBlob, "recorded-video.webm");

    try {
      // Determine API URL based on environment
      let backendUrl;
      if (window.location.hostname === "localhost") {
        backendUrl = "http://localhost:8000/predict";
      } else {
        // For Replit environment - use the full URL with port 8000
        backendUrl = `${window.location.protocol}//${window.location.hostname}:8000/predict`;
      }

      console.log("Sending request to:", backendUrl);
      
      const response = await fetch(backendUrl, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (data.error) {
        alert(data.error);
        setDetectedEmotion("");
      } else {
        setDetectedEmotion(data.predicted_emotion);
        console.log("Emotion Scores:", data.scores);
      }
    } catch (err) {
      console.error("Error uploading video for emotion:", err);
      setDetectedEmotion("");
    }
  };

  // ------------------------------------------------------------------
  // 2) Upload for Transcription => POST /transcribe
  //    We only reset "transcribedText" so old emotion & LLM remain
  // ------------------------------------------------------------------
  const handleUploadForTranscription = async () => {
    if (!recordedBlob) {
      alert("No recorded video available.");
      return;
    }
    setTranscribedText("(loading...)");

    const formData = new FormData();
    formData.append("file", recordedBlob, "recorded-video.webm");

    try {
      const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (data.detail) {
        alert(data.detail);
        setTranscribedText("");
      } else if (data.transcription) {
        setTranscribedText(data.transcription);
        console.log("Transcription:", data.transcription);
      }
    } catch (err) {
      console.error("Error uploading video for transcription:", err);
      setTranscribedText("");
    }
  };

  // ------------------------------------------------------------------
  // 3) Upload for Everything => POST /process_all
  //    Overwrites all fields, but only if successful
  // ------------------------------------------------------------------
  const handleProcessAll = async () => {
    if (!recordedBlob) {
      alert("No recorded video available.");
      return;
    }

    // Optional: show loading text
    setDetectedEmotion("(loading...)");
    setTranscribedText("(loading...)");
    setLlmResponse("(loading...)");

    const formData = new FormData();
    formData.append("file", recordedBlob, "recorded-video.webm");

    try {
      const response = await fetch("http://localhost:8000/process_all", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        // We can handle 4xx/5xx errors more explicitly:
        const errorText = await response.text();
        console.error("Server error at /process_all:", errorText);
        alert(`Error: ${errorText}`);
        // Clear the loading states, or revert them
        setDetectedEmotion("");
        setTranscribedText("");
        setLlmResponse("");
        return;
      }

      const data = await response.json();

      if (data.error) {
        alert(data.error);
        // Clear the loading states
        setDetectedEmotion("");
        setTranscribedText("");
        setLlmResponse("");
      } else {
        setDetectedEmotion(data.predicted_emotion);
        setTranscribedText(data.transcription);
        setLlmResponse(data.llm_response);
        console.log("All results:", data);
      }
    } catch (err) {
      console.error("Error uploading video for combined processing:", err);
      // Clear the loading states
      setDetectedEmotion("");
      setTranscribedText("");
      setLlmResponse("");
    }
  };

  return (
    <div className="video-container">
      <h1>Record Your Emotion &amp; Transcription</h1>

      <video ref={videoRef} autoPlay muted playsInline />

      <div className="button-container">
        <button 
          className={`action-button start ${isRecording ? 'disabled' : ''}`} 
          onClick={startRecording} 
          disabled={isRecording}>
          <i className="fas fa-video"></i> Start Recording
        </button>
        <button 
          className={`action-button stop ${!isRecording ? 'disabled' : ''}`} 
          onClick={stopRecording} 
          disabled={!isRecording}>
          <i className="fas fa-stop-circle"></i> Stop Recording
        </button>
      </div>

      <div>
        <button onClick={handleUploadForEmotion} disabled={!recordedBlob}>
          Upload for Emotion
        </button>
        <button onClick={handleUploadForTranscription} disabled={!recordedBlob}>
          Upload for Transcription
        </button>
        <button onClick={handleProcessAll} disabled={!recordedBlob}>
          Upload for Emotion + Transcription + LLM
        </button>
      </div>

      {/* Show the results */}
      <p>
        <strong>Detected Emotion:</strong> {detectedEmotion}
      </p>
      <p>
        <strong>Transcribed Text:</strong> {transcribedText}
      </p>
      <p>
        <strong>LLM Response:</strong> {llmResponse}
      </p>
    </div>
  );
};

export default VideoControl;
