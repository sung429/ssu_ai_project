from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data="plate_dataset.yaml", 
    device=[0,1,2,3], 
    epochs=50, 
    batch=512,
    imgsz=320,
    project='plate_train',
    multi_scale=True, 
    amp=True)