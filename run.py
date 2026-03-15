from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import cv2
import os
import base64

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
app = FastAPI()

# Load your YOLO model here. 
# NOTE: You MUST upload your trained model weights (e.g., 'best.pt') to Azure alongside this run.py file!
try:
    model = YOLO('best.pt') 
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/snapshot")
def snapshot(rtsp_url: str = Query(...)):
    if model is None:
         return JSONResponse(content={"error": "AI Model not loaded on server"}, status_code=500)

    # 1. Capture the frame
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return JSONResponse(content={"error": "Cannot open stream"}, status_code=502)
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return JSONResponse(content={"error": "Cannot read frame"}, status_code=502)

    # 2. Run YOLO detection to find the meters
    results = model(frame)
    meters_dict = {}
    
    # Extract bounding boxes from the results
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
    # 3. Crop and encode each detected meter
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Cut the image using numpy slicing [startY:endY, startX:endX]
        cropped_meter = frame[y1:y2, x1:x2]
        
        # Convert the cropped image to a base64 string
        _, buffer = cv2.imencode('.jpg', cropped_meter)
        meter_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Add to our dictionary
        meters_dict[f"meter_{i+1}"] = meter_base64
        
    # Handle the case where the AI doesn't see any meters
    if not meters_dict:
        return JSONResponse(content={"message": "No meters detected in this frame"}, status_code=404)

    # 4. Return the dictionary (FastAPI automatically formats this as JSON)
    return meters_dict

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8585)
