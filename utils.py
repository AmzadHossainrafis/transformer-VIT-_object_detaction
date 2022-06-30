import shutil
import keras
import os

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
