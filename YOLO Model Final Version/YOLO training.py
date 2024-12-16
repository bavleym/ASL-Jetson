from ultralytics import YOLO
import torch
import os

# Create the directory if it doesn't exist
save_dir = r"C:\Users\Gaiaf\OneDrive - CSULB\Documents\CECS 490 B\YOLO_Exports"
os.makedirs(save_dir, exist_ok=True)

if __name__ == '__main__':
    print(torch.cuda.is_available())  # Should print True if GPU is available
    print(torch.cuda.get_device_name(0))  # Prints the name of the GPU

    # Load the YOLO model
    model = YOLO('yolov8n.pt')  # Use a pretrained YOLOv8 model

    # Train the model
    model.train(data=r"C:\Users\Gaiaf\OneDrive - CSULB\Documents\CECS 490 B\American Sign Language Letters.v1-v1.yolov8\data.yaml",
                epochs=50, imgsz=640, device=0)

    # Evaluate the model
    metrics = model.val(data=r"C:\Users\Gaiaf\OneDrive - CSULB\Documents\CECS 490 B\American Sign Language Letters.v1-v1.yolov8\data.yaml")
    # Print the metrics
    print(metrics)

    # Export the model with a specific save directory
    exported_model_path = model.export(format='onnx', imgsz=640, save_dir=r"C:\Users\Gaiaf\OneDrive - CSULB\Documents\CECS 490 B\YOLO_Exports")
    print(f"Model exported to: {exported_model_path}")

