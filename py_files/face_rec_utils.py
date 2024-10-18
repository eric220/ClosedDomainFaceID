import gradio as gr
import numpy as np
import cv2 as cv2
import pickle
import face_recognition
from statistics import mode

with open('data/processed/lfw_names.pickle', 'rb') as handle:
    names = pickle.load(handle)
    
with open('data/processed/lfw_encodings2.pickle', 'rb') as handle:
    face_encodings = pickle.load(handle)
    
def get_test_encoding(t_frame):
    t_encodings = []
    rgb = cv2.cvtColor(t_frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    if len(boxes) != 0:
        encoding = face_recognition.face_encodings(rgb, boxes)
        t_encodings.append(encoding)
    return boxes, t_encodings

#def matches(known_encodings, unknown_encoding):
def matches(unknown_encoding):
    threshold = .5
    matches = ['NOT SURE']
    ####change from known_encodings
    for i, k_e in enumerate(face_encodings):
        dist = np.linalg.norm(np.array(k_e) - np.array(unknown_encoding[0]), axis = 1)
        if dist[0] < threshold:
            matches.append(names[i])
    try:
        match = mode(matches)
    except:
        match = n[0]
    return match