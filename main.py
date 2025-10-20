# =====================================================================================
# AI Assignment: Footfall Counter using Computer Vision
#
# main.py
#
# Description:
# This script sets up a FastAPI web server to provide an API endpoint for the
# footfall counting logic. It handles video uploads, calls the processing function,
# and serves the resulting video and heatmap as static files.
#
# Author: Akhand Pratap Shukla
# Date: 21/10/2025
# =====================================================================================

import uvicorn
import os
import uuid
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

# Import the core logic from our other file
from counter_logic import run_footfall_analysis

# Create FastAPI app instance
app = FastAPI(title="Footfall Counter API", description="Upload a video to count people and generate a heatmap.")

# --- Define directories ---
UPLOADS_DIR = "uploads"
OUTPUTS_DIR = "outputs"
# Create directories if they don't exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Mount the 'outputs' directory so that files inside it can be served
# This means a file like 'outputs/heatmap.png' will be accessible at 'http://localhost:8000/outputs/heatmap.png'
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/")
def read_root():
    """A simple root endpoint to check if the API is running."""
    return {"message": "Welcome to the Footfall Counter API. Send a POST request to /process-video/ to analyze a video."}


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    """
    Accepts a video file, processes it, and returns the counts and output file paths.
    """
    try:
        # Generate a unique filename to prevent conflicts
        unique_id = uuid.uuid4().hex
        _, file_extension = os.path.splitext(file.filename)
        input_video_path = os.path.join(UPLOADS_DIR, f"{unique_id}{file_extension}")

        # Save the uploaded video file temporarily
        with open(input_video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Call the synchronous processing function
        # FastAPI is smart and runs this in an external threadpool to not block the server
        results = run_footfall_analysis(video_path=input_video_path, output_dir=OUTPUTS_DIR)

        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])

        # Clean up the uploaded file
        os.remove(input_video_path)
        
        # Return the results in a JSON response
        return {
            "message": "Video processed successfully!",
            "counts": {
                "in": results["in_count"],
                "out": results["out_count"]
            },
            # Return web-accessible URLs for the output files
            "results": {
                "processed_video_url": f"/outputs/{os.path.basename(results['processed_video_path'])}",
                "heatmap_image_url": f"/outputs/{os.path.basename(results['heatmap_image_path'])}"
            }
        }
    except Exception as e:
        # Handle potential errors during processing
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# This block allows running the server directly with `python main.py`
if __name__ == '__main__':
    print("Starting FastAPI server...")
    print("Access the API docs at http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)