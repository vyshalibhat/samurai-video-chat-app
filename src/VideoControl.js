import React, { useRef, useState } from "react";
import "./VideoControl.css";

const VideoControl = () => {
  const videoRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [detectedEmotion, setDetectedEmotion] = useState("");

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
    formData.append("file", recordedBlob, "recorded-video.webm");

    try {
      const response = await fetch("http://0.0.0.0:8000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.error) {
        alert(data.error);
      } else {
        setDetectedEmotion(data.predicted_emotion);
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
    </div>
  );
};

export default VideoControl;
