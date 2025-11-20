import cv2
import dlib
from scipy.spatial import distance as dist

# Load face landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# eye landmark indexes
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def detect_blink(frame, thresh=0.20):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return False

    for face in faces:
        shape = predictor(gray, face)
        shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

        left_eye = [shape[i] for i in LEFT_EYE]
        right_eye = [shape[i] for i in RIGHT_EYE]

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0

        if ear < thresh:  # eyes closed
            return True

    return False
