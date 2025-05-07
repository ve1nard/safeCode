import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        tar.extractall("/tmp/unpack")

    # get the file name of the file we want to extract
    file_name = os.path.basename(__file__).split(".")[0]

    # get the absolute path of the file we want to extract
    file_path = os.path.join("/tmp/unpack", file_name)

    # if the file exists, remove it
    if os.path.exists(file_path):
        os.remove(file_path)

    # copy the file to the file_path
    shutil.copyfile(__file__, file_path)
