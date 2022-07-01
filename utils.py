import shutil
import keras
import os
import yaml
import numpy as np

images, targets = [], []
path_images = "/101_ObjectCategories/airplanes/"
path_annot = "/Annotations/Airplanes_Side_2/"

def Download_data():
    '''
    Download the data+annotarion from the url and extract it in 
    directory /101_ObjectCategories/airplanes/

    return: sorted list of images and annotatons 
    '''

    path_to_downloaded_file = keras.utils.get_file(
        fname="caltech_101_zipped",
        origin="https://data.caltech.edu/tindfiles/serve/e41f5188-0b32-41fa-801b-d1e840915e80/",
        extract=True,
        archive_format="zip",  # downloaded file format
        cache_dir="/",  # cache and extract in current directory
    )

    # Extracting tar files found inside main zip file
    
    shutil.unpack_archive("/datasets/caltech-101/101_ObjectCategories.tar.gz", "/")
    shutil.unpack_archive("/datasets/caltech-101/Annotations.tar", "/")

    # list of paths to images and annotations
    image_paths = [
        f for f in os.listdir(path_images) if os.path.isfile(os.path.join(path_images, f))
    ]
    annot_paths = [
        f for f in os.listdir(path_annot) if os.path.isfile(os.path.join(path_annot, f))
    ]

    return image_paths.sort(),annot_paths.sort()


def read_yaml(path='config.yaml'):
    """
    Reads the yaml file and returns the data in a dictionary.
    :param path: The path to the yaml file.
    :return: The data in the yaml file.
    """
    with open(path, 'r') as stream:
        data_loaded = yaml.load(stream, Loader=yaml.FullLoader)
    return data_loaded

for i in range(0, len(annot_paths)):
    # Access bounding box coordinates
    annot = scipy.io.loadmat(path_annot + annot_paths[i])["box_coord"][0]

    top_left_x, top_left_y = annot[2], annot[0]
    bottom_right_x, bottom_right_y = annot[3], annot[1]

    image = keras.utils.load_img(
        path_images + image_paths[i],
    )
    (w, h) = image.size[:2]

    # resize train set images
    if i < int(len(annot_paths) * 0.8):
        # resize image if it is for training dataset
        image = image.resize((image_size, image_size))

    # convert image to array and append to list
    images.append(keras.utils.img_to_array(image))

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

config= read_yaml()