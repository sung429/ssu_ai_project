import os
import glob
import json
import cv2
import tqdm
import shutil
import numpy as np

image_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/images/'
label_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/original_labels/labels/merged/'
target_label_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/datasets/'

label_list = glob.glob(label_path+'*/*/*.json')

print(len(label_list))

def xyxy_to_xywh(bbox, image):
    # 좌 상단 x
    xl = bbox[0][0]
    # 좌 상단 y
    yt = bbox[0][1]
    # 우 하단 x
    xr = bbox[1][0]
    # 우 하단 y
    yb = bbox[1][1]

    # 이미지 정보 확인 (높이, 너비, 채널)
    h, w, c = image.shape

    # yolo의 좌표는 바운딩 박스의 x축 중심, y축 중심, 박스의 너비, 박스의 높이로 구성되고 값은 0~1 값으로 되어있음
    width = (xr - xl)/w
    height = (yb - yt)/h
    xc = ((xl + xr)/2)/w
    yc = ((yt + yb)/2)/h

    return xc, yc, width, height
i = 0
for label_file in tqdm.tqdm(label_list):
    with open(label_file, 'r') as f:
        json_file = json.load(f)
        # print(json_file)
        # save_directory = 'valid' if i // 10 == 8 else 'test' if i // 10 == 9 else 'train'
        image_exist = os.path.exists(image_path+json_file['imagePath'])
        image_file = image_path+json_file['imagePath']
        plate_bbox = json_file['plate']['bbox']
        # 한글 경로 문제 때문에 cv2.imread 사용 불가
        # image = cv2.imread(image_file)
        try:
            img_array = np.fromfile(image_file, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except:
            continue
        # print(xyxy_to_xywh(plate_bbox, image))
        try:
            if not os.path.exists(f"{target_label_path}/raw/labels/{image_file.split('/')[-1].split('.')[0]}.txt"):
                with open(f"{target_label_path}/raw/labels/{image_file.split('/')[-1].split('.')[0]}.txt", 'w') as lf:
                    lf.write(f'{0} {xyxy_to_xywh(plate_bbox, image)[0]} {xyxy_to_xywh(plate_bbox, image)[1]} {xyxy_to_xywh(plate_bbox, image)[2]} {xyxy_to_xywh(plate_bbox, image)[3]}\n')

            else:
                with open(f"{target_label_path}/raw/labels/{image_file.split('/')[-1].split('.')[0]}.txt", 'a') as lf:
                    lf.write(f'{0} {xyxy_to_xywh(plate_bbox, image)[0]} {xyxy_to_xywh(plate_bbox, image)[1]} {xyxy_to_xywh(plate_bbox, image)[2]} {xyxy_to_xywh(plate_bbox, image)[3]}\n')

            shutil.copy(image_file, f"{target_label_path}/raw/images/{image_file.split('/')[-1]}")
            i += 1
        except:
            continue






    