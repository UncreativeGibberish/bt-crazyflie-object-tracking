from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo26n")  
    #model = RTDETR("rtdetr-l.pt") # For training RTDETR models
    results = model.train(
        data="config.yaml",          
        epochs=150,
        imgsz=320,                   
        batch=0.85,                    
        device=0,                    
        patience=50,                 
        project="runs/train",
        name="car_detector_320",
        exist_ok=True
    )