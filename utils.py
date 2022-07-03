import os
import yaml
import numpy as np
import scipy.io
import tensorflow as tf

def read_yaml(path='config.yaml'):
    """
    Reads the yaml file and returns the data in a dictionary.
    :param path: The path to the yaml file.
    :return: The data in the yaml file.
    """
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)

    return data_loaded


config= read_yaml()



def data_preproses():


    images, targets = [], []
    path_images = config['path_images']
    path_annot = config['path_annot']

    image_paths = [
    f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))
    ]
    annot_paths = [
        f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))
    ]

    image_paths.sort()
    annot_paths.sort()

    image_size = config['image_size']  # resize input images to this size

    images, targets = [], []

    # loop over the annotations and images, preprocess them and store in lists
    for i in range(0, len(annot_paths)):
        # Access bounding box coordinates
        annot = scipy.io.loadmat(path_annot + annot_paths[i])["box_coord"][0]

        top_left_x, top_left_y = annot[2], annot[0]
        bottom_right_x, bottom_right_y = annot[3], annot[1]

        image = tf.keras.utils.load_img(
            path_images + image_paths[i],
        )
        (w, h) = image.size[:2]

        # resize train set images
        if i < int(len(annot_paths) * 0.8):
            # resize image if it is for training dataset
            image = image.resize((image_size, image_size))

        # convert image to array and append to list
        images.append(tf.keras.utils.img_to_array(image))

        # apply relative scaling to bounding boxes as per given image and append to list
        targets.append(
            (
                float(top_left_x) / w,
                float(top_left_y) / h,
                float(bottom_right_x) / w,
                float(bottom_right_y) / h,
            )
        )

    (x_train), (y_train) = (
        np.asarray(images[: int(len(images) * 0.8)]),
        np.asarray(targets[: int(len(targets) * 0.8)]),
    )
    (x_test), (y_test) = (
        np.asarray(images[int(len(images) * 0.8) :]),
        np.asarray(targets[int(len(targets) * 0.8) :]),

    )
    return x_train, y_train, x_test, y_test

m,n,o,p=data_preproses()
