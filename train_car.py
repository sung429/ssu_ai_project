from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data="car_dataset.yaml", 
    device=[0,1,2,3], 
    epochs=20, 
    batch=256,
    project='car_train',
    multi_scale=True, 
    amp=True)