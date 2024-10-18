import numpy as np
import glob
import cv2 as cv2
import dlib
import face_recognition
import pickle

def get_file_names():
    '''return names of files in directory'''
    roots = glob.glob('../data/raw2/train/*')
    all_files = [glob.glob(x + '/*') for x in roots]
    file_list = [item for sublist in all_files for item in sublist]
    return file_list

def encodings_names(f_list):
    '''gets encodings and names for all files in directory'''
    face_encodings = []
    names = []
    for f in f_list[:]:
        name = f.split('/')[4]
        image = cv2.imread(f)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if len(boxes) != 0:
            encodings = face_recognition.face_encodings(rgb, boxes)
            face_encodings.append(encodings)
            names.append(name)
    return face_encodings, names

def save_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def main():        
    file_list = get_file_names()
    face_encodings, names = encodings_names(file_list)
    save_data(names, '../data/processed/names2.pickle')
    save_data(face_encodings, '../data/processed/face_encodings2.pickle')
    
if __name__ == '__main__':
    main()