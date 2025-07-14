from ultralytics import YOLO 

model = YOLO('yolov8n.pt')

results = model.predict('input_video/video de prueba.mp4',save=True)
print(results[0])
print('=====================================')
for box in results[0].boxes:
    print(box)
    
