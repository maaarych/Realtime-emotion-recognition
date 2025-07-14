import cv2
from deepface import DeepFace

# Start video capture
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame for emotions
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Get dominant emotion and display it
        dominant_emotion = result[0]['dominant_emotion']
        cv2.putText(frame, dominant_emotion, (50, 50), font, 1, (0, 255, 0), 2)
    
    except Exception as e:
        print(f"Error: {e}")

    # Show the frame
    cv2.imshow("Face Emotion Recognition", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
