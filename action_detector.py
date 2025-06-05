import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hands Up Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Hands Up Detection", 960, 720)  # Set width and height as you like

while True:
    success, img = cap.read()
    if not success:
        break

    # Convert the BGR image to RGB before processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the image and detect pose
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        left_wrist_y = landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_wrist_y = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        if left_wrist_y < left_shoulder_y and right_wrist_y >= right_shoulder_y:
            status = "Left Hand Up"
        elif right_wrist_y < right_shoulder_y and left_wrist_y >= left_shoulder_y:
            status = "Right Hand Up"
        elif left_wrist_y < left_shoulder_y and right_wrist_y < right_shoulder_y:
            status = "Both Hands Up"
        else:
            status = "Hands Down"
        cv2.putText(img, status, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Draw pose landmarks on the image
        # mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the image
    cv2.imshow("Hands Up Detection", img)

    # Press ESC key to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
