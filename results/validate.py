from ultralytics import YOLO
import os

def validate_model():
   
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    
    model_path = os.path.join(ROOT_DIR, "models", "blur_finetuned.pt")
    data_yaml = os.path.join(ROOT_DIR, "dataset", "data.yaml")

    print("Loading model...")
    model = YOLO(model_path)

    print("Running validation on TEST dataset...\n")
    metrics = model.val(
        data=data_yaml,
        split="test",
        verbose=True
    )

    results = {
        "Precision": float(metrics.box.mp),
        "Recall": float(metrics.box.mr),
        "mAP@50": float(metrics.box.map50),
        "mAP@50-95": float(metrics.box.map)
    }

    print("\nValidation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results



if __name__ == "__main__":
    validate_model()
