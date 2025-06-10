import os
import glob
import shutil
import tqdm

base_dir = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/datasets'
plate_base_dir = '/media2/dataset/103.자동차_차종-연식-번호판_인식용_데이터/original/original/datasets/plate'

raw_dir = os.path.join(base_dir, 'raw')
train_dir = os.path.join(base_dir, 'train')
valid_dir = os.path.join(base_dir, 'valid')
test_dir = os.path.join(base_dir, 'test')

plate_raw_dir = os.path.join(plate_base_dir, 'raw')
plate_train_dir = os.path.join(plate_base_dir, 'train')
plate_valid_dir = os.path.join(plate_base_dir, 'valid')
plate_test_dir = os.path.join(plate_base_dir, 'test')

os.makedirs(train_dir+'/images', exist_ok=True)
os.makedirs(train_dir+'/labels', exist_ok=True)
os.makedirs(valid_dir+'/images', exist_ok=True)
os.makedirs(valid_dir+'/labels', exist_ok=True)
os.makedirs(test_dir+'/images', exist_ok=True)
os.makedirs(test_dir+'/labels', exist_ok=True)

os.makedirs(plate_train_dir+'/images', exist_ok=True)
os.makedirs(plate_train_dir+'/labels', exist_ok=True)
os.makedirs(plate_valid_dir+'/images', exist_ok=True)
os.makedirs(plate_valid_dir+'/labels', exist_ok=True)
os.makedirs(plate_test_dir+'/images', exist_ok=True)
os.makedirs(plate_test_dir+'/labels', exist_ok=True)

raw_image_list = glob.glob(os.path.join(raw_dir, 'images', '*.jpg'))
plate_raw_image_list = glob.glob(os.path.join(plate_raw_dir, 'images', '*.jpg'))


print(len(raw_image_list))

image_list = glob.glob(os.path.join(raw_dir, 'images', '*.jpg'))
label_list = glob.glob(os.path.join(raw_dir, 'labels', '*.txt'))
plate_image_list = glob.glob(os.path.join(plate_raw_dir, 'images', '*.jpg'))
plate_label_list = glob.glob(os.path.join(plate_raw_dir, 'labels', '*.txt'))

# 이미지 라벨 일치 확인
# for image_path in tqdm.tqdm(image_list):
#     label_path = image_path.replace('images', 'labels').replace('.jpg', '.txt')
#     if label_path not in label_list:
#         print(image_path)

for i, image_path in tqdm.tqdm(enumerate(image_list)):
    if i % 10 == 1:
        shutil.move(image_path, image_path.replace('raw', 'valid'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'valid').replace('.jpg', '.txt').replace('images', 'labels'))
    elif i % 10 == 2:
        shutil.move(image_path, image_path.replace('raw', 'test'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'test').replace('.jpg', '.txt').replace('images', 'labels'))
    else:
        shutil.move(image_path, image_path.replace('raw', 'train'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'train').replace('.jpg', '.txt').replace('images', 'labels'))

for i, image_path in tqdm.tqdm(enumerate(plate_image_list)):
    if i % 10 == 1:
        shutil.move(image_path, image_path.replace('raw', 'valid'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'valid').replace('.jpg', '.txt').replace('images', 'labels'))
    elif i % 10 == 2:
        shutil.move(image_path, image_path.replace('raw', 'test'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'test').replace('.jpg', '.txt').replace('images', 'labels'))
    else:
        shutil.move(image_path, image_path.replace('raw', 'train'))
        shutil.move(image_path.replace('.jpg', '.txt').replace('images', 'labels'), image_path.replace('raw', 'train').replace('.jpg', '.txt').replace('images', 'labels'))