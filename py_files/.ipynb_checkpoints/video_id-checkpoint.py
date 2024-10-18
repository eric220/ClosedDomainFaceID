import numpy as np
import glob
import cv2 as cv2
import dlib
import face_recognition
from statistics import mode
import pickle

with open('../data/processed/lfw_names.pickle', 'rb') as handle:
    names = pickle.load(handle)
    
with open('../data/processed/lfw_encodings.pickle', 'rb') as handle:
    face_encodings = pickle.load(handle)
    
def get_test_encoding(t_frame):
    t_encodings = []
    rgb = cv2.cvtColor(t_frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if len(boxes) != 0:
        encoding = face_recognition.face_encodings(rgb, boxes)
        t_encodings.append(encoding)
    return boxes, t_encodings

def matches(known_encodings, unknown_encoding):
    distances = []
    for k_e in known_encodings:
        dist = np.linalg.norm(np.array(k_e) - np.array(unknown_encoding[0]), axis = 1)
        distances.append(dist)
    n = ['Unknown']
    for i, j in enumerate(distances):
        if j[0] < .6:
            n.append(names[i])
    try:
        m = mode(n)
    except:
        m = n[0]
    return m

def main():
    cap= cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    color = (255, 0, 0)
    thickness = 2
    while True:
        cv2.startWindowThread()
        ret, frame = cap.read()
        boxes, t_encodings = get_test_encoding(frame)
        if len(boxes) > 0:
            for i, t in enumerate(t_encodings):
                t_name = matches(face_encodings, t)
                (y, r, b, x) = boxes[i]
                cv2.rectangle(frame, (x, y), (r, b), (255, 0, 0), 2)
                cv2.putText(frame, '{}'.format(t_name), (x-10, y-10), font, fontScale, color, thickness, cv2.LINE_AA)
            #t_name = matches(face_encodings, t_encodings)
            #for (y, r, b, x) in boxes:
                #cv2.rectangle(frame, (x, y), (r, b), (255, 0, 0), 5)
                #cv2.putText(frame, '{}'.format(t_name), (x-10, y-10), font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('Video', frame)
        if cv2.waitKey(25) == 13:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == '__main__':
    main()