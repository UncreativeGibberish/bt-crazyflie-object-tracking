from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("best.pt")  
    results = model.val(
        data="G:/pythonscripts/newdata.yolo26/config.yaml",
        split="val",
        imgsz=320,
        batch=0.85,
        conf=0.25,
        iou=0.7,
        plots=True,
        save_json=True,
        device=0
    )

    
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    print(f"mAP50-95 : {results.results_dict['metrics/mAP50-95(B)']:.4f}")
    print(f"mAP50    : {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall   : {results.results_dict['metrics/recall(B)']:.4f}")
    print("="*50)
