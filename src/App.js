import React, { useEffect, useRef, useState } from "react";
import "./App.css";
import MainPage from './MainPage';

function App() {
  const videoRef = useRef(null);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    async function setupCamera() {
      try {
        // Log that we're attempting to access the webcam
        console.log("Requesting webcam access...");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        console.log("Webcam stream obtained:", stream);
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          // Wait for the video metadata to load to confirm that the feed is ready
          videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setStreaming(true);
            console.log("Video playback started");
          };
        }
      } catch (err) {
        console.error("Error accessing media devices:", err);
        setError("Error accessing webcam: " + err.message);
      }
    }

    // Call the async function
    setupCamera();
  }, []);

  return (
    <div className="App">
      <div>
            <MainPage />
        </div>
    </div>
  );
}

export default App;