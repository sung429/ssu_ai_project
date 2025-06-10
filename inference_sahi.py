from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# model = YOLO("runs/detect/train4/weights/best.pt")

# results = model('test_dataset/vlcsnap-2025-06-02-13h22m06s352.png')

# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     keypoints = result.keypoints
#     probs = result.probs
#     obb = result.obb
#     # result.show()
#     result.save(filename="result.jpg")


detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="runs/detect/train7/weights/best.pt",
    confidence_threshold=0.3,
    device="cuda"
)

result = get_sliced_prediction('test_dataset/vlcsnap-2025-06-02-13h22m06s352.png', detection_model)

result.export_visuals(export_dir="demo_data/")