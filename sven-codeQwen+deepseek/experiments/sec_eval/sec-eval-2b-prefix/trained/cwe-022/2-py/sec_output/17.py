import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # and the directories should be created if they don't exist
        tar.extractall("/tmp/unpack")

    # now we can work with the files
    with open("/tmp/unpack/file1") as f:
        print(f.read())

    # we can also work with the directories
    with open("/tmp/unpack/dir1") as f:
        print(f.read())
