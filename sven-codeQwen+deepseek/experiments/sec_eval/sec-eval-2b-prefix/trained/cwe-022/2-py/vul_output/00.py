import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        tar.extractall("/tmp/unpack")

    # move the files under /tmp/unpack to the destination
    os.rename("/tmp/unpack/", "/home/ubuntu/")
