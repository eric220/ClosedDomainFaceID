import glob
import numpy as np
import cv2 as cv2
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def resize_img(timg):
    width = int(300/timg.shape[1] * timg.shape[1])
    height = int(400/timg.shape[0] * timg.shape[0])
    dim = (width, height)
    resized = cv2.resize(timg, dim, interpolation = cv2.INTER_AREA)
    return resized

#load image and get name
def get_image_name():
    try:
        file_name = glob.glob('../data/temporary/*')
        img = cv2.imread(str(file_name[0])) 
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        subject_name = input('Name of person')
    except:
        print('Image Not Found')
    return subject_name, im_rgb

def mutate_img(t_img, name):
    '''Mutates and saves img to file'''
    datagen = ImageDataGenerator(
        rotation_range= 10,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')

    x = img_to_array(t_img)  
    x = x.reshape((1,) + x.shape)  
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='../data/processed/temporary_mutated_images/{}'.format(name),
                              save_prefix='{}'.format(name), save_format='jpeg'):
        i += 1
        if i > 5:
            break 

def create_directory(name):
    try:
        filename = Path('../data/processed/temporary_mutated_images/{}'.format(name)).mkdir()
    except:
        print('Folder Already Exists')

def main():
    '''takes in image from temporary folder, mutates it and stores to temporary_mutated images folder'''
    name, img = get_image_name()
    resized_img = resize_img(img)
    create_directory(name)
    mutate_img(resized_img, name)

if __name__ == '__main__':
    main()