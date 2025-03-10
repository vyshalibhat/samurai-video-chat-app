// VideoControl.js

import React, { useRef, useState } from "react";
import "./VideoControl.css";

const VideoControl = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const [transcribedText, setTranscribedText] = useState(""); // For transcript
  const [llmResponse, setLlmResponse] = useState("");         // For LLM output

  // 1) Start recording using webcam (with audio and video)
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;

      // Priority list of MIME types
      const mimeTypes = [
        "video/webm;codecs=vp9",
        "video/mp4",
        "video/avi",
        "video/webm;codecs=vp8",
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
        console.warn("No specified MIME types supported, using default");
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
        // For debugging: log the preview URL (copy-paste it to test playback)
        const previewUrl = URL.createObjectURL(blob);
        console.log("Preview URL:", previewUrl);
        // Stop the camera stream
        stream.getTracks().forEach((track) => track.stop());
        videoRef.current.srcObject = null;
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("Error accessing media devices:", err);
    }
  };

  // 2) Stop recording
  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // 3) Upload the recorded file to the backend and update states with the response
  const uploadVideo = async () => {
    if (!recordedBlob) {
      alert("No recorded video available.");
      return;
    }

    const formData = new FormData();
    formData.append("file", recordedBlob, "recorded-video.webm");

    // Determine API URL based on environment
    let backendUrl;
    if (window.location.hostname === "localhost") {
      backendUrl = "http://localhost:8000/predict";
    } else {
      backendUrl = `${window.location.protocol}//${window.location.hostname}:8000/predict`;
    }
    console.log("Sending request to:", backendUrl);

    try {
      const response = await fetch(backendUrl, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server responded with status: ${response.status}`);
      }

      const responseText = await response.text();
      console.log("Raw response:", responseText);

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (jsonErr) {
        console.error("Failed to parse JSON response:", jsonErr);
        alert("Invalid response format from server");
        return;
      }

      if (data.error) {
        console.error("Server error:", data.error);
        alert(`Error from server: ${data.error}`);
      } else {
        setDetectedEmotion(data.predicted_emotion || "Unknown");
        setTranscribedText(data.transcribed_text || ""); // Set transcript
        setLlmResponse(data.llm_response || "");         // Set LLM response
        console.log("Emotion scores:", data.scores);
      }
    } catch (err) {
      console.error("Error uploading video:", err);
      alert(`Error: ${err.message}`);
    }
  };

  return (
    <div className="video-container">
      <h1>Record Your Emotion</h1>
      <video ref={videoRef} autoPlay muted playsInline />
      <div className="button-container">
        <button onClick={startRecording} disabled={isRecording}>
          Start Recording
        </button>
        <button onClick={stopRecording} disabled={!isRecording}>
          Stop Recording
        </button>
        <button onClick={uploadVideo} disabled={!recordedBlob}>
          Upload Video
        </button>
      </div>
      {detectedEmotion && (
        <p className="emotion">
          Detected Emotion: <strong>{detectedEmotion}</strong>
        </p>
      )}
      {transcribedText && (
        <p className="transcript">
          Transcribed Speech: <strong>{transcribedText}</strong>
        </p>
      )}
      {llmResponse && (
        <p className="llm">
          LLM Response: <strong>{llmResponse}</strong>
        </p>
      )}
    </div>
  );
};

export default VideoControl;
