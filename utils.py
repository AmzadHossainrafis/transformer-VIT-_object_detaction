import shutil
import keras

path_images = "/101_ObjectCategories/airplanes/"
path_annot = "/Annotations/Airplanes_Side_2/"

def Download_data():

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