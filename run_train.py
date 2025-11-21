from ultralytics import YOLO
import pathlib

proj = pathlib.Path(r"D:\My Downloads\AstroShield_project")
data_yaml = proj / "data" / "dataset.yaml"

model = YOLO(str(proj / "yolov8n.pt"))  # starting weights

model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=640,
    project=str(proj / "runs"),
    name="train_experiment",
    exist_ok=True
)
