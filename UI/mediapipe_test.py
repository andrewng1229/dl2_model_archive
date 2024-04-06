import mediapipe as mp
import cv2
from mediapipe.tasks.python import vision

model_path = "../Mediapipe/naruto_hand_gestures.task"
recognizer = vision.GestureRecognizer.create_from_model_path(model_path)
recognizer_landmark = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = recognizer_landmark.process(frame_rgb)
    cv2.imwrite('temp_frame.jpg', frame_rgb)
    image = mp.Image.create_from_file('temp_frame.jpg')

    recognition_result = recognizer.recognize(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            pass

    if results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            if hand_label == "Left":
              cv2.putText(frame, f'Hand: {hand_label}', (45, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            else:
              cv2.putText(frame, f'Hand: {hand_label}', (45, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            if recognition_result.gestures:
                top_gesture = recognition_result.gestures[0][0]
                cv2.putText(frame, f'Gesture recognized: {top_gesture.category_name} ({top_gesture.score})',
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('cv2',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()