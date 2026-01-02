from ultralytics import YOLO

if __name__ == '__main__':  # This line is absolutely mandatory on Windows
    # Load the model
    model = YOLO('yolov8n.pt') 

    # Train the model
    results = model.train(
        data='gate_dataset/data.yaml', 
        epochs=50,
        imgsz=[480, 640],   # [Height, Width]
        
        # --- CRITICAL FIXES FOR WINDOWS ---
        batch=8,            # Lower batch size (RTX 3050 has 4GB VRAM, 16 might be too high)
        workers=1,          # Reduce workers from 8 to 1 to save System RAM
        # ----------------------------------
        
        name='flare_model_fixed'
    )