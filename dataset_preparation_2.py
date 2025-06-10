import os
import glob
import json
import cv2
import tqdm
import shutil
import numpy as np
import multiprocessing as mp

image_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/image/'
label_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/original_labels/labels/merged/'
target_label_path = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/datasets/'

# 필요한 디렉터리 생성
os.makedirs(f"{target_label_path}/raw/images", exist_ok=True)
os.makedirs(f"{target_label_path}/raw/labels", exist_ok=True)
os.makedirs(f"{target_label_path}/plate/raw/images", exist_ok=True)
os.makedirs(f"{target_label_path}/plate/raw/labels", exist_ok=True)

label_list = glob.glob(label_path+'*/*/*.json')

print(f"총 레이블 파일 수: {len(label_list)}")

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

def process_label(label_file):
    """각 레이블 파일을 처리하는 함수"""
    try:
        with open(label_file, 'r') as f:
            json_file = json.load(f)
            
        # 이미지 파일 경로
        image_file = image_path + json_file['imagePath']
        
        # plate 정보가 있는지 확인
        if 'plate' not in json_file.keys():
            return {'status': 'error', 'file': label_file, 'reason': 'no plate info'}
            
        car_bbox = json_file['car']['bbox']
        plate_bbox = json_file['plate']['bbox']
        resized_plate_bbox = []
        # print("car_bbox")
        # print(car_bbox)
        # print("plate_bbox")
        # print(plate_bbox)
        # print(plate_bbox[0] - car_bbox[0])
        # print(plate_bbox[1] - car_bbox[0])
        for plate in plate_bbox:
            resized_plate_bbox.append([plate[0] - car_bbox[0][0], plate[1] - car_bbox[0][1]])
        # plate_bbox = resized_plate_bbox
        # resized_plate_bbox = [[plate[0] - car[0], plate[1] - car[0]] for car, plate in zip(car_bbox, plate_bbox)]
        
        # 한글 경로 문제 때문에 cv2.imread 사용 불가
        try:
            img_array = np.fromfile(image_file, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            car_image = image.copy()[int(car_bbox[0][1]):int(car_bbox[1][1]), int(car_bbox[0][0]):int(car_bbox[1][0])]
            if image is None:
                raise Exception("이미지 디코딩 실패")
        except Exception as e:
            return {'status': 'error', 'file': label_file, 'reason': f'image load failed: {str(e)}'}
        
        # YOLO 좌표 변환
        try:
            xc, yc, width, height = xyxy_to_xywh(car_bbox, image)
            label_xc, label_yc, label_width, label_height = xyxy_to_xywh(resized_plate_bbox, car_image)
        except Exception as e:
            return {'status': 'error', 'file': label_file, 'reason': f'coordinate conversion failed: {str(e)}'}
        
        # 레이블 파일 작성
        label_file_path = f"{target_label_path}/raw/labels/{image_file.split('/')[-1].split('.')[0]}.txt"
        plate_label_file_path = f"{target_label_path}/plate/raw/labels/{image_file.split('/')[-1].split('.')[0]}.txt"
        try:
            if not os.path.exists(label_file_path):
                with open(label_file_path, 'w') as lf:
                    lf.write(f'0 {xc} {yc} {width} {height}\n')
            else:
                with open(label_file_path, 'a') as lf:
                    lf.write(f'0 {xc} {yc} {width} {height}\n')

            if not os.path.exists(plate_label_file_path):
                with open(plate_label_file_path, 'w') as lf:
                    lf.write(f'0 {label_xc} {label_yc} {label_width} {label_height}\n')
            else:
                with open(plate_label_file_path, 'a') as lf:
                    lf.write(f'0 {label_xc} {label_yc} {label_width} {label_height}\n')
                    
            # 이미지 파일 복사
            shutil.copy(image_file, f"{target_label_path}/raw/images/{image_file.split('/')[-1]}")
            cv2.imwrite(f"{target_label_path}/plate/raw/images/{image_file.split('/')[-1]}", car_image)
            # shutil.copy(image_file, f"{target_label_path}/plate/raw/images/{image_file.split('/')[-1]}")
            
            return {'status': 'success', 'file': label_file}
            
        except Exception as e:
            return {'status': 'error', 'file': label_file, 'reason': f'file operation failed: {str(e)}'}
            
    except Exception as e:
        return {'status': 'error', 'file': label_file, 'reason': f'general error: {str(e)}'}

if __name__ == '__main__':
    # multiprocessing 사용
    with mp.Pool(processes=8) as pool:
        results = list(tqdm.tqdm(pool.imap(process_label, label_list), total=len(label_list)))
    
    # 결과 분석
    success_count = 0
    error_count = 0
    except_list = []
    
    for result in results:
        if result['status'] == 'success':
            success_count += 1
        else:
            error_count += 1
            except_list.append(result)
    
    print(f"처리 완료: 성공 {success_count}, 실패 {error_count}")
    
    # 예외 목록 저장
    with open("except_list.txt", "w") as f:
        for except_item in except_list:
            f.write(f"{except_item['file']} - {except_item['reason']}\n")
    
    print(f"예외 목록이 except_list.txt에 저장되었습니다.")




    