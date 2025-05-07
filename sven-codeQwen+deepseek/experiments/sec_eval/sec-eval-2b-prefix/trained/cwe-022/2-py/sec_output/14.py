import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just use the tarfile directly
        tar.extractall("/tmp/unpack")

    # remove the tar file
    os.remove("archive.tar.gz")
