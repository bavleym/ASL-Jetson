import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import mediapipe as mp
import torchvision.models as models
import torch.nn as nn

# Load the ResNet18 model
model = models.resnet18()

# Modify the final classification layer to match 28 classes (A-Z, Nothing, Space)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Linear(128, 28)  # 28 classes in total
)

# Load the saved model weights
model_path = "fivek_resnet18_finetuned_model.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
model.to(device)
model.eval()

# Define the transformation for live camera input (match the training transformation)
transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the label map for A-Z, Nothing, and Space
label_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J",
    10: "K", 11: "L", 12: "M", 13: "N", 14: "Nothing", 15: "O", 16: "P", 17: "Q",
    18: "R", 19: "S", 20: "Space", 21: "T", 22: "U", 23: "V", 24: "W", 25: "X",
    26: "Y", 27: "Z"
}

# Initialize MediaPipe hand detection
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture webcam video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands_detector.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                hand_coords = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in hand_landmarks.landmark]
                x_min = max(min([coord[0] for coord in hand_coords]) - 20, 0)
                y_min = max(min([coord[1] for coord in hand_coords]) - 20, 0)
                x_max = min(max([coord[0] for coord in hand_coords]) + 20, frame.shape[1])
                y_max = min(max([coord[1] for coord in hand_coords]) + 20, frame.shape[0])
                
                hand_region = frame[y_min:y_max, x_min:x_max]

                # Darken the background outside the hand region
                mask = np.zeros_like(frame)
                mask[y_min:y_max, x_min:x_max] = hand_region
                frame_darkened = cv2.addWeighted(frame, 0.5, mask, 0.5, 0)

                # Convert the hand region to RGB (match the input transformation)
                pil_image_rgb = Image.fromarray(hand_region)

                # Apply the transformations (resize and convert to tensor)
                input_tensor = transform(pil_image_rgb).unsqueeze(0).to(device)

                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = F.softmax(output, dim=1)
                    predicted_class = torch.argmax(pred, dim=1).item()
                    confidence = torch.max(pred).item() * 100

                predicted_label = label_map[predicted_class]

                # Draw the bounding box and display the label with confidence
                cv2.rectangle(frame_darkened, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
                cv2.putText(frame_darkened, f"Predicted: {predicted_label} ({confidence:.2f}%)", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        else:
            frame_darkened = frame

        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame_darkened)

    # Press 'q' to exit the video window
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
