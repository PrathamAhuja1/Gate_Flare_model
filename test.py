from ultralytics import YOLO

model = YOLO('runs_gate_model/detect/gate_model_fixed/weights/best.pt')

# source='0' uses the default webcam. 
# Replace '0' with a video file path ('test_video.mp4') to test on recorded video.
results = model.predict(source='output_gate.mp4', show=True, conf=0.5)