import cv2
import mediapipe as mp

# Initialize MediaPipe solutions
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

# Using the enums makes the code more readable
FINGER_TIPS = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
               mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP

def count_fingers(hand_landmarks, hand_label):
    """
    Counts the number of open fingers for a given hand.
    The logic is corrected to handle both left and right hands.
    """
    fingers_open = 0
    
    # Check four fingers (Index, Middle, Ring, Pinky)
    for tip_id in FINGER_TIPS:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers_open += 1

    # Check Thumb
    if hand_label == "Right":
        if hand_landmarks.landmark[THUMB_TIP].x < hand_landmarks.landmark[THUMB_TIP - 1].x:
            fingers_open += 1
    elif hand_label == "Left":
        if hand_landmarks.landmark[THUMB_TIP].x > hand_landmarks.landmark[THUMB_TIP - 1].x:
            fingers_open += 1

    return fingers_open, 5 - fingers_open

def get_hand_position_from_pose(pose_landmarks):
    """
    Determines if hands are 'UP' or 'DOWN' based on wrist and shoulder landmarks.
    """
    try:
        left_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        left_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
        left_up = left_wrist_y < left_shoulder_y
    except:
        left_up = False

    try:
        right_wrist_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        right_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        right_up = right_wrist_y < right_shoulder_y
    except:
        right_up = False

    return left_up, right_up

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
     mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        frame.flags.writeable = False
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results_pose = pose.process(frame_rgb)
        results_hands = hands.process(frame_rgb)
        
        frame.flags.writeable = True

        
        posture_status = "Not detected"
        rh_open, rh_closed = "N/A", "N/A"
        lh_open, lh_closed = "N/A", "N/A"
        right_hand_pos = "DOWN"
        left_hand_pos = "DOWN"

        
        if results_pose.pose_landmarks:
            posture_status = "Normal"
            left_up, right_up = get_hand_position_from_pose(results_pose.pose_landmarks)
            right_hand_pos = 'UP' if right_up else 'DOWN'
            left_hand_pos = 'UP' if left_up else 'DOWN'
           # mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        
        if results_hands.multi_hand_landmarks:
            for hand_landmarks, hand_handedness in zip(results_hands.multi_hand_landmarks, results_hands.multi_handedness):
                label = hand_handedness.classification[0].label
                open_f, closed_f = count_fingers(hand_landmarks, label)
                if label == "Right":
                    rh_open, rh_closed = open_f, closed_f
                elif label == "Left":
                    lh_open, lh_closed = open_f, closed_f
              #  mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        
        overlay = [
            "[Status Panel]", f"Right Hand: {right_hand_pos}",
            f"  Fingers Open: {rh_open}", f"  Fingers Closed: {rh_closed}",
            "", f"Left Hand: {left_hand_pos}",
            f"  Fingers Open: {lh_open}", f"  Fingers Closed: {lh_closed}",
            "", f"Posture: {posture_status}"
        ]
        y_start, line_height = 25, 20
        for i, line in enumerate(overlay):
            y = y_start + i * line_height
            cv2.rectangle(frame, (5, y - line_height + 5), (200, y + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Hand and Pose Status Panel", cv2.resize(frame, None, fx=1.5, fy=1.5))


        if cv2.waitKey(5) & 0xFF == 27:
            break


cap.release()
cv2.destroyAllWindows()