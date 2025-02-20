import cv2
import mediapipe as mp
import math

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def check_posture(landmarks):
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    nose = landmarks[mp_pose.PoseLandmark.NOSE]

    shoulder_alignment = abs(left_shoulder.y - right_shoulder.y)
    nose_to_shoulder = (left_shoulder.y + right_shoulder.y) / 2 - nose.y

    bad_alignment_threshold = 0.1
    slouching_threshold = -0.1

    if shoulder_alignment > bad_alignment_threshold or nose_to_shoulder < slouching_threshold:
        return "Bad Posture"
    return "Good Posture"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    posture_status = "Good Posture"
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        posture_status = check_posture(results.pose_landmarks.landmark)

    color = (0, 255, 0) if posture_status == "Good Posture" else (0, 0, 255)
    cv2.putText(
        frame, 
        posture_status, 
        (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        color, 
        2
    )

    cv2.imshow("Posture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
