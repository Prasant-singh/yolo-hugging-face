from ultralytics import YOLO
import cv2
import os




model=YOLO("yolov8n.pt")

def model_predict(img,filename,save_dir):
    results = model.predict(source=img)
    for result in results:
        classes=result.names
        for box in result.boxes:
            if box.conf>0.7:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_id = int(box.cls[0])
                class_name =classes[class_id]
                cv2.putText(img, f'{class_name} {box.conf[0]:.2f}', (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    processed_file=f'processed_{filename}'
    processed_filepath=os.path.join(save_dir, processed_file)
    cv2.imwrite(processed_filepath, img)
    return processed_filepath