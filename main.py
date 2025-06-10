import glob
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

car_model = YOLO("car_train/train5/weights/best.pt")
plate_model = YOLO("plate_train/train7/weights/best.pt")
reader = easyocr.Reader(['ko']) 

test_data = glob.glob('test_dataset/*.png')

def draw_box(img, box, label):
    # img = cv2.imread(img)
    color = (0, 255, 0)
    thickness = 2
    font_scale=0.6
    cv2.rectangle(img, (box[0],box[1]), (box[2],box[3]), color, thickness)
    cv2.putText(img, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
    
    return img

for test_image in test_data:
    results = car_model(test_image)

    for result in results:
        car_boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)[0]
        orig_img = result.orig_img
        car_crop_img = orig_img.copy()[car_boxes[1]:car_boxes[3],car_boxes[0]:car_boxes[2]]
        plate_results = plate_model(car_crop_img)
        for result in plate_results:
            if result.boxes.xyxy.shape[0] != 0:
                print(result.boxes)
                print(result.boxes.xyxy.shape)
                plate_boxes = result.boxes.xyxy.cpu().numpy().astype(np.int32)[0]
                car_crop_img = result.orig_img[plate_boxes[1]:plate_boxes[3],plate_boxes[0]:plate_boxes[2]]
                
                ocr_result = reader.readtext(car_crop_img)
                print(ocr_result)
                ocr_value = 'Fail' if len(ocr_result)==0 else ocr_result[0][1]
                print(ocr_value)
                print(type(ocr_value))
                # result.show()
                orig_img = draw_box(orig_img, car_boxes, ocr_value)
        
        cv2.imwrite(f"result_data_plate/{test_image.split('/')[-1]}", orig_img)
            
            # result.save(filename=f"result_data_plate/{test_image.split('/')[-1]}")


    