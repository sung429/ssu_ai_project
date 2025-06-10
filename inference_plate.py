import glob
from ultralytics import YOLO

model = YOLO("car_train/train5/weights/best.pt")
plate_model = YOLO("plate_train/train7/weights/best.pt")

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
        result.save_crop(save_dir="result_data", file_name=f"{test_image.split('/')[-1]}")

test_data = glob.glob('result_data/car/*.jpg')

for test_image in test_data:
    results = plate_model(test_image)

    for result in results:
        boxes = result.boxes
        masks = result.masks
        keypoints = result.keypoints
        probs = result.probs
        obb = result.obb
        # result.show()
        result.save(filename=f"result_data_plate/{test_image.split('/')[-1]}")