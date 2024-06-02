from ultralytics import YOLO
model = YOLO("yolov9e.pt")  # load a pretrained model (recommended for training)
# model = YOLO("runs/detect/train/weights/epoch54.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(project="YOLOv8", data="til.yaml", epochs=80, device=[5, 6], imgsz=1600,
                      single_cls=True, save=True, batch=6, patience=5, deterministic=False,
                      save_period=1, cos_lr=True, workers=8, cache=True,
                      plots=True, amp=True, resume=False, mosaic=True)
model.val()