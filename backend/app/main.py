
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to SamurAI API"}

@app.post("/analyze-video")
async def analyze_video():
    # This will be implemented later to process the video
    # and return emotion analysis
    return {
        "detected_emotion": "Happy",
        "confidence": 0.92,
        "recommendation": "You seem to be in a good mood!"
    }
