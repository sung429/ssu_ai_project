import glob
from ultralytics import YOLO

model = YOLO("car_train/train5/weights/best.pt")

test_data = glob.glob('test_dataset/*.png')

for test_image in test_data:
    results = model(test_image)

    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        # result.show()
        result.save(filename=f"result_data/{test_image.split('/')[-1]}")