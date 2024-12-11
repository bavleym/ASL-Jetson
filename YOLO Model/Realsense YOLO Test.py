import torch
from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import logging

# Reduce logs
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load the trained YOLO model
model_path = r"C:\Users\Bavley\OneDrive - csulb\Documents\CECS 490 B\best.onnx"
model = YOLO(model_path)

# Define label mapping
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Initialize RealSense pipeline
pipeline = rs.pipeline()
align = rs.align(rs.stream.color)  # Align depth to color

# Configure the RealSense pipeline
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

try:
    # Start RealSense streaming
    pipeline.start(config)
    print("Streaming started. Press 'q' to exit.")

    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert RealSense frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply a color map to the depth image
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # Perform YOLO inference on the color image
        results = model.predict(source=color_image, conf=0.5, device='cpu', imgsz=640, verbose=False)

        # Decode YOLO predictions
        predictions = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
        decoded_predictions = [labels[int(cls)] for cls in predictions]

        # Draw YOLO predictions
        for det, pred_label in zip(results[0].boxes.data.cpu().numpy(), decoded_predictions):
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(color_image, pred_label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Process the color image with MediaPipe for hand tracking
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        results_hands = hands.process(rgb_image)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                # Draw MediaPipe hand landmarks
                mp_drawing.draw_landmarks(
                    color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Extract hand coordinates and calculate bounding box
                hand_coords = [(int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0]))
                               for lm in hand_landmarks.landmark]
                x_min = min([coord[0] for coord in hand_coords])
                y_min = min([coord[1] for coord in hand_coords])
                x_max = max([coord[0] for coord in hand_coords])
                y_max = max([coord[1] for coord in hand_coords])

                # Add padding to bounding box
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(color_image.shape[1], x_max + padding)
                y_max = min(color_image.shape[0], y_max + padding)

                # Draw bounding box around the hand on both color and depth colormaps
                cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.rectangle(depth_colormap, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw landmarks on the depth colormap
                for coord in hand_coords:
                    cv2.circle(depth_colormap, coord, radius=5, color=(255, 255, 255), thickness=-1)

        # Display depth information for the center of the bounding box
        if results_hands.multi_hand_landmarks:
            center_x = (x_min + x_max) // 2
            center_y = (y_min + y_max) // 2
            depth_value = depth_frame.get_distance(center_x, center_y)
            cv2.putText(depth_colormap, f"Depth: {depth_value:.2f}m",
                        (10, depth_colormap.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Stack the color and depth views side by side
        combined_image = np.hstack((color_image, depth_colormap))

        # Display the combined image
        cv2.imshow("YOLO + MediaPipe + Depth", combined_image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    hands.close()
