from ultralytics import YOLO
import os, json, time

project_root = r"D:\My Downloads\AstroShield_project"
weights = os.path.join(project_root, "best.pt")
data_yaml = os.path.join(project_root, "data.yaml")

print("Using model:", weights)
print("Using dataset config:", data_yaml)
print("Running validation... Please wait.\n")

model = YOLO(weights)
results = model.val(data=data_yaml, imgsz=640)

# Extract metrics safely
metrics = {}
box = getattr(results, "box", None)
if box is not None:
    metrics["mAP50"] = getattr(box, "map50", None) or getattr(box, "map", None)
    metrics["mAP50_95"] = getattr(box, "map50_95", None)
    metrics["precision"] = getattr(box, "precision", None)
    metrics["recall"] = getattr(box, "recall", None)

print("Extracted Metrics:")
print(json.dumps(metrics, indent=2))

output_path = os.path.join(project_root, f"validation_metrics_{int(time.time())}.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nSaved metrics to:", output_path)
