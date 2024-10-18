import glob
import numpy as np
import cv2 as cv2
from pathlib import Path
import pickle
import face_recognition
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def resize_img(timg):
    """
    Resizes an image to a target width and height.

    Parameters:
        timg (numpy.ndarray): The input image to be resized.

    Returns:
        numpy.ndarray: The resized image.

    """
    width = int(300/timg.shape[1] * timg.shape[1])#THIS DOES NOT DO WHAT YOU THINK IT DOES
    height = int(400/timg.shape[0] * timg.shape[0])#THIS DOES NOT DO WHAT YOU THINK IT DOES
    dim = (width, height)
    resized = cv2.resize(timg, dim, interpolation = cv2.INTER_AREA)
    return resized

def get_image_name():
    """
    Retrieves the name of a person associated with an image.

    Returns:
        tuple: A tuple containing the subject name and the image in RGB format.

    Raises:
        Exception: If an image is not found.

    """
    try:
        file_name = glob.glob('data/temporary/*')#../removed from path
        img = cv2.imread(str(file_name[0])) 
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        subject_name = input('Name of person')
    except:
        print('Image Not Found')
    return subject_name, im_rgb

def create_directory(name):
    """
    Creates a directory for storing processed temporary mutated images.

    Parameters:
        name (str): The name of the directory to be created.

    Raises:
        Exception: If the directory already exists.

    """
    try:
        filename = Path('data/processed/temporary_mutated_images/{}'.format(name)).mkdir() #../ removed from path
    except:
        print('Folder Already Exists')
        
def delete_directory(f: Path):
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            delete_directory(child)
        f.rmdir()
        
def mutate_img(t_img, name):
    """
    Mutates and saves an image to a file.

    Parameters:
        t_img (PIL.Image.Image): The input image to be mutated.
        name (str): The name of the image file and directory.

    """
    datagen = ImageDataGenerator(
        rotation_range= 10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.0,
        zoom_range=0.0,
        horizontal_flip=False,
        fill_mode='nearest')

    x = img_to_array(t_img)  
    x = x.reshape((1,) + x.shape)  
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='data/processed/temporary_mutated_images/{}'.format(name),#../ removed from path
                              save_prefix='{}'.format(name),
                              save_format='jpeg'):
        i += 1
        if i > 5:
            break 
            
def save_data(data, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def get_names_encodings(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
    
def encodings_names(f_list):
    """
    Extracts face encodings and corresponding names from a list of image file paths.

    Parameters:
        f_list (list): A list of image file paths.

    Returns:
        tuple: A tuple containing the extracted face encodings and corresponding names.

    """
    t_face_encodings = []
    t_names = []
    for f in f_list[:]:
        name = f.split('/')[3]
        image = cv2.imread(f)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        if len(boxes) != 0:
            encodings = face_recognition.face_encodings(rgb, boxes)
            t_face_encodings.append(encodings)
            t_names.append(name)
    return t_face_encodings, t_names

def main():
    name, img = get_image_name()
    resized_img = resize_img(img)
    create_directory(name)
    mutate_img(resized_img, name)
    names = get_names_encodings('../data/processed/lfw_names.pickle')
    encodings = get_names_encodings('../data/processed/lfw_encodings2.pickle')
    file_list = glob.glob('../data/processed/temporary_mutated_images/{}/*'.format(name))
    t_encodings, t_names = encodings_names(file_list)
    for t in t_names:
        names.append(t)
    for e in t_encodings:
        encodings.append(e)
    assert len(names) == len(encodings)
    save_data(names, '../data/processed/lfw_names.pickle')
    save_data(encodings, '../data/processed/lfw_encodings.pickle')
    delete_directory(Path('../data/processed/temporary_mutated_images/{}'.format(name))) 
    
if __name__ == '__main__':
    main()