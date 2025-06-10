from ultralytics import YOLO

model = YOLO('yolo11n.pt')

model.train(
    data="dataset.yaml", 
    device=[0,1,2,3], 
    epochs=20, 
    batch=56, 
    multi_scale=False, 
    imgsz=1920, 
    rect=True, amp=True)