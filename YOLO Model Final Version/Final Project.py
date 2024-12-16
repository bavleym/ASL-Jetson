import torch
from ultralytics import YOLO
import cv2
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import logging
import time

# Suppress logging from libraries
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Detect GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {'GPU: ' + torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

# Load YOLO model
model_path = r"C:\Users\Gaiaf\OneDrive - csulb\Documents\CECS 490 B\best.onnx"
model = YOLO(model_path)  # Load ONNX model

# Define label mapping (A-Z)
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

# Configure RealSense camera pipeline
pipeline = rs.pipeline()
align = rs.align(rs.stream.color)  # Align depth to color
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Depth filtering thresholds (meters)
min_depth = 0.2
max_depth = 1.2

# Gesture tracking variables
gesture_start_time = None
current_gesture = None
last_gesture_time = time.time()  # Initialize to the current time

# Word construction variables
word_buffer = ""
max_line_length = 20  # Characters per line in word display
word_timeout = 2  # Timeout in seconds to separate words
clear_buffer_timeout = 7  # Clear buffer after 7 seconds of inactivity


def wrap_text(text, max_length):
    """
    Wrap text to fit within a specified maximum line length.
    """
    return '\n'.join([text[i:i + max_length] for i in range(0, len(text), max_length)])


def detect_space_gesture(hand_landmarks):
    """
    Detect a 'Space' gesture by checking if all fingers are extended.
    """
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP,
    ]
    finger_pips = [
        mp_hands.HandLandmark.THUMB_IP,
        mp_hands.HandLandmark.INDEX_FINGER_PIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
        mp_hands.HandLandmark.RING_FINGER_PIP,
        mp_hands.HandLandmark.PINKY_PIP,
    ]

    # Ensure all fingertips are above their corresponding PIP joints
    return all(
        hand_landmarks.landmark[finger_tips[i]].y < hand_landmarks.landmark[finger_pips[i]].y
        for i in range(len(finger_tips))
    )


def visualize_depth(color_image, depth_image):
    """
    Combine color image with depth visualization for side-by-side display.
    """
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    return np.hstack((color_image, depth_colormap))


def process_frame(color_image, depth_image):
    """
    Process the current frame: detect hands, track gestures, and run YOLO inference.
    """
    global word_buffer, gesture_start_time, current_gesture, last_gesture_time

    # YOLO inference on the full frame
    results = model.predict(source=color_image, conf=0.75, device=device, imgsz=640, verbose=False)

    # Extract predictions and confidences
    predictions = results[0].boxes.cls.cpu().numpy() if results[0].boxes.cls is not None else []
    confidences = results[0].boxes.conf.cpu().numpy() if results[0].boxes.conf is not None else []

    # Update the last gesture time if a gesture is detected
    if predictions.size > 0:
        last_gesture_time = time.time()  # Update the time since a gesture was detected

        # Take the first detected letter and its confidence
        detected_gesture = labels[int(predictions[0])]
        confidence = confidences[0]

        if detected_gesture == current_gesture and confidence >= 0.75:
            if gesture_start_time is None:
                gesture_start_time = time.time()
            elif time.time() - gesture_start_time >= 0.5:
                word_buffer += detected_gesture
                current_gesture = None
                gesture_start_time = None
        else:
            current_gesture = detected_gesture
            gesture_start_time = None
    else:
        current_gesture = None
        gesture_start_time = None

    # Process MediaPipe for hand gestures
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    results_hands = hands.process(rgb_image)

    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            if detect_space_gesture(hand_landmarks):
                if gesture_start_time is None:
                    gesture_start_time = time.time()
                elif time.time() - gesture_start_time >= 0.5:
                    word_buffer += "_"  # Append space
                    gesture_start_time = None
                    current_gesture = None

            # Draw hand landmarks
            mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Timeout logic to separate words
    if time.time() - last_gesture_time > word_timeout:
        word_buffer += " "  # Append a space to separate words

    # Buffer clear logic
    time_since_last_gesture = time.time() - last_gesture_time
    if time_since_last_gesture > clear_buffer_timeout:
        word_buffer = ""  # Clear the buffer
        last_gesture_time = time.time()  # Reset the last gesture time to prevent immediate clearing

    # Wrap the word buffer for display
    wrapped_word_buffer = wrap_text(word_buffer, max_line_length)

    # Display the word buffer in the top-left corner
    y0, dy = 50, 30
    for i, line in enumerate(wrapped_word_buffer.split('\n')):
        cv2.putText(color_image, line, (10, y0 + i * dy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display countdown in the bottom-right corner
    buffer_clear_countdown = max(0, int(clear_buffer_timeout - time_since_last_gesture))
    height, width, _ = color_image.shape
    cv2.putText(color_image, f"{buffer_clear_countdown}s", (width - 100, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Draw YOLO predictions
    for det, pred_label in zip(results[0].boxes.data.cpu().numpy(), predictions):
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(color_image, f"{labels[int(cls)]} ({conf:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return color_image


try:
    # Start RealSense streaming
    pipeline.start(config)
    print("Streaming started. Press 'q' to exit.")

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Convert RealSense frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        color_image = cv2.rotate(color_image, cv2.ROTATE_180)
        depth_image = cv2.rotate(depth_image, cv2.ROTATE_180)

        # Process and display the current frame
        processed_frame = process_frame(color_image, depth_image)
        combined_frame = visualize_depth(processed_frame, depth_image)
        cv2.imshow("YOLO + MediaPipe + Depth Visualization", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
