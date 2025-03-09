// VideoControl.js

import React, { useRef, useState } from "react";
import "./VideoControl.css";

const VideoControl = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState("");
  const [transcribedText, setTranscribedText] = useState(""); // NEW state for transcript
  const [llmResponse, setLlmResponse] = useState("");         // NEW state for LLM response

  // 1) Attempt multiple MIME types in order
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });
      videoRef.current.srcObject = stream;

      // Priority list of mime types
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
          // Debug: create a preview URL and open it in a new tab
        const previewUrl = URL.createObjectURL(blob);
        console.log("Preview URL:", previewUrl);
        window.open(previewUrl, "_blank");
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

  // 2) Upload the recorded file
  const uploadVideo = async () => {
    if (!recordedBlob) {
      alert("No recorded video available.");
      return;
    }

    const formData = new FormData();
    // Using "recorded-video.webm" as filename; your backend will handle both webm and mp4
    formData.append("file", recordedBlob, "recorded-video.webm");

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        alert(data.error);
      } else {
        // Update state with emotion, transcription, and LLM response
        setDetectedEmotion(data.predicted_emotion);
        setTranscribedText(data.transcribed_text); // NEW: Set transcript
        setLlmResponse(data.llm_response);           // NEW: Set LLM reply
        console.log("Scores:", data.scores);
      }
    } catch (err) {
      console.error("Error uploading video:", err);
    }
  };

  return (
    <div className="video-container">
      <h1>Record Your Emotion</h1>

      <video ref={videoRef} autoPlay muted playsInline />

      <div>
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
        <p>
          Detected Emotion: <strong>{detectedEmotion}</strong>
        </p>
      )}
      {transcribedText && (
        <p>
          Transcribed Speech: <strong>{transcribedText}</strong>
        </p>
      )}
      {llmResponse && (
        <p>
          LLM Response: <strong>{llmResponse}</strong>
        </p>
      )}
    </div>
  );
};

export default VideoControl;
