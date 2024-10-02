import mediapipe as mp
import cv2
import time

hands = mp.solutions.hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Front camera
cap.set(cv2.CAP_PROP_FPS, 60)  # Set frame rate to 15 FPS

nice_guy_start_frame = None
frame_counter = 1
last_saved_frame = None

def fingers_up(hand_landmarks, is_right_hand):
    # Get the landmark coordinates
    landmarks = hand_landmarks.landmark
    
    # Initialize count for fingers that are up
    fingers = []
    
    # Initialize word
    word = ""
    word_detected = False

    # Thumb: handle left and right hands separately
    if is_right_hand:  # Assuming it's a right hand
        # Right hand: thumb tip is to the left of the base joint
        fingers.append(1 if landmarks[4].x < landmarks[2].x else 0)
        
        # Create condition for detecting word
        if (landmarks[4].y < landmarks[2].y and landmarks[5].y and landmarks[9].y and landmarks [13].y and landmarks [17].y) and (landmarks[8].x > landmarks[6].x and landmarks[12].x > landmarks[10].x and landmarks[16].x > landmarks[14].x and landmarks[20].x > landmarks[18].x):
            word = "Good"
            word_detected = True
        elif landmarks[4].y > landmarks[0].y and landmarks[1].y and landmarks [2].y and landmarks [3].y:
            word = "Bad"
            word_detected = True
        elif landmarks[12].y < (landmarks[8].y > landmarks[6].y) and (landmarks[16].y > landmarks[14].y) and (landmarks[20].y > landmarks[18].y) and (landmarks[12].y < landmarks[10].y):
            word = "You're a Nice Guy"
            word_detected = True

    else:
        # Left hand: thumb tip is to the right of the base joint
        fingers.append(1 if landmarks[4].x > landmarks[2].x else 0)
        word_detected = False

    # For the other four fingers, check if the tip is higher (in the y-axis) than the lower joint
    # Index finger
    fingers.append(1 if landmarks[8].y < landmarks[6].y else 0)  # Tip (8) is above joint (6)
    # Middle finger
    fingers.append(1 if landmarks[12].y < landmarks[10].y else 0)
    # Ring finger
    fingers.append(1 if landmarks[16].y < landmarks[14].y else 0)
    # Pinky finger
    fingers.append(1 if landmarks[20].y < landmarks[18].y else 0)

    return fingers, word, word_detected

def fingers_down(hand_landmarks, is_right_hand):
    # Get the landmark coordinates
    landmarks = hand_landmarks.landmark
    
    # Initialize count for fingers that are up
    fingers = []

    # Thumb: handle left and right hands separately
    if is_right_hand:  # Assuming it's a right hand
        # Right hand: thumb tip is to the left of the base joint
        fingers.append(1 if landmarks[2].x < landmarks[4].x else 0)
    else:
        # Left hand: thumb tip is to the right of the base joint
        fingers.append(1 if landmarks[2].x > landmarks[4].x else 0)

    # For the other four fingers, check if the tip is higher (in the y-axis) than the lower joint
    # Index finger
    fingers.append(1 if landmarks[6].y < landmarks[8].y else 0)  # Tip (8) is above joint (6)
    # Middle finger
    fingers.append(1 if landmarks[10].y < landmarks[12].y else 0)
    # Ring finger
    fingers.append(1 if landmarks[14].y < landmarks[16].y else 0)
    # Pinky finger
    fingers.append(1 if landmarks[18].y < landmarks[20].y else 0)

    return fingers

while True:
    success, image = cap.read()
    #print("Camera success: ", success, end='\r')

    if not success:
        break

    # Flip the image horizontally for easier reading
    image = cv2.flip(image, 1)

    # Process the image with MediaPipe Hands
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
    results = hands.process(image_rgb)

    total_fingers_up = 0
    total_fingers_down = 0
    word = ""
    word_detected = False

    # Draw hand landmarks and connections
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            # Determine if the hand is right or left
            is_right_hand = handedness.classification[0].label == "Right"

            # Check how many fingers are up
            finger_status, word, word_detected = fingers_up(hand_landmarks, is_right_hand)
            #left_finger_status = lfingers_up(hand_landmarks)
            fingers_up_count = sum(finger_status)  # Total number of fingers up
            
            # Check how many fingers are down
            fingers_status = fingers_down(hand_landmarks, is_right_hand)
            fingers_down_count = sum(fingers_status)

            total_fingers_up += fingers_up_count
            total_fingers_down += fingers_down_count
            
             # Handle the word detection and display logic
            if word_detected and word == "You're a Nice Guy":
                if nice_guy_start_time is None:  # Start tracking time if the word is detected
                    nice_guy_start_time = time.time()
            else:
                nice_guy_start_time = None  # Reset the timer if the word disappears

            # If the word is detected for more than 2 seconds, save the frame
            if nice_guy_start_time is not None:
                elapsed_time = time.time() - nice_guy_start_time
                if elapsed_time > .1:
                    current_time = time.time()
                    # Ensure 2 seconds have passed since the last save
                    if last_saved_frame is None or (current_time - last_saved_frame > 1):
                        # Save the frame with a unique indexed filename
                        filename = f'nice_guy_frame_{frame_counter}.jpg'
                        cv2.imwrite(filename, image)
                        frame_counter += 1  # Increment the frame counter
                        last_saved_frame = current_time  # Update the last saved time
            
            if word_detected == True:
                if word =="Good":
                    word = "Good"
                elif word == "Bad":
                    word = "Bad"

        # Display the detected word if present
        if word_detected:
            cv2.putText(image, f'Word: {word}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            # Display the count on the image
            cv2.putText(image, f'Fingers Up: {total_fingers_up}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Fingers Down: {total_fingers_down}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the processed image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(40) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()