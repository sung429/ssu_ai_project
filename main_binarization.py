import glob
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from utils import draw_box

car_model = YOLO("car_train/train5/weights/best.pt")
plate_model = YOLO("plate_train/train10/weights/best.pt")
reader = easyocr.Reader(['ko'], recog_network='custom_2nd') 

test_data = glob.glob('test_dataset/*.png')
test_data = sorted(test_data)


def export_csv(results):
    with open('trained_binarization.csv', 'w', encoding='cp949') as f:
        f.writelines(results)


def binarization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary


results_ocr = []
for test_image in test_data:
    results = car_model(test_image)

    for result in results:
        car_boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)[0]
        orig_img = result.orig_img
        car_crop_img = orig_img.copy()[car_boxes[1]:car_boxes[3],car_boxes[0]:car_boxes[2]]
        plate_results = plate_model(car_crop_img)
        for result in plate_results:
            if result.boxes.xyxy.shape[0] != 0:
                # print(result.boxes)
                # print(result.boxes.xyxy.shape)
                plate_boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)[0]
                plate_crop_img = result.orig_img[plate_boxes[1]:plate_boxes[3],plate_boxes[0]:plate_boxes[2]]
                plate_crop_img = binarization(plate_crop_img)
                cv2.imwrite(f"result_data_plate_bina/{test_image.split('/')[-1]}", plate_crop_img)
                ocr_result = reader.readtext(plate_crop_img)
                # print(ocr_result)
                ocr_value = 'Fail' if len(ocr_result)==0 else ocr_result[0][1]
                results_ocr.append(f"{test_image.split('/')[-1]}, {ocr_value}\n")
                # print(ocr_value)
                # print(type(ocr_value))
                # result.show()
                orig_img = draw_box(orig_img, car_boxes, ocr_value)
            else:
                results_ocr.append(f"{test_image.split('/')[-1]}, None\n")
        
        cv2.imwrite(f"result_data_plate_binarization/{test_image.split('/')[-1]}", orig_img)
    export_csv(results_ocr) 
            # result.save(filename=f"result_data_plate/{test_image.split('/')[-1]}")


    