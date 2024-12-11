import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import cv2

def main():
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # Create RealSense pipeline
    pipeline = rs.pipeline()
    align = rs.align(rs.stream.color)  # Align depth to color

    # Configure the pipeline
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Moderate resolution for performance
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Moderate resolution for performance

    try:
        # Start streaming
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

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply a color map to the depth image
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
            )

            # Process the color image with MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw MediaPipe hand landmarks on the color image
                    mp_drawing.draw_landmarks(
                        color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )

                    # Extract hand coordinates
                    hand_coords = [(int(lm.x * color_image.shape[1]), int(lm.y * color_image.shape[0]))
                                   for lm in hand_landmarks.landmark]

                    # Calculate bounding box
                    x_min = min([coord[0] for coord in hand_coords])
                    y_min = min([coord[1] for coord in hand_coords])
                    x_max = max([coord[0] for coord in hand_coords])
                    y_max = max([coord[1] for coord in hand_coords])

                    # Expand the bounding box slightly
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(color_image.shape[1], x_max + padding)
                    y_max = min(color_image.shape[0], y_max + padding)

                    # Draw bounding box on the color image
                    cv2.rectangle(color_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Draw bounding box and landmarks on the depth colormap
                    cv2.rectangle(depth_colormap, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    for coord in hand_coords:
                        cv2.circle(depth_colormap, coord, radius=5, color=(255, 255, 255), thickness=-1)

            # Stack both images side by side
            combined_image = np.hstack((color_image, depth_colormap))

            # Display the combined image
            cv2.imshow('Color and Depth with MediaPipe Overlay', combined_image)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Streaming stopped.")
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()
