from fastapi import FastAPI, Query
from fastapi.responses import Response
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
app = FastAPI()

@app.get("/snapshot")
def snapshot(rtsp_url: str = Query(...)):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        return Response(content='{"error":"Cannot open stream"}', status_code=502, media_type="application/json")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return Response(content='{"error":"Cannot read frame"}', status_code=502, media_type="application/json")
    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8585)
