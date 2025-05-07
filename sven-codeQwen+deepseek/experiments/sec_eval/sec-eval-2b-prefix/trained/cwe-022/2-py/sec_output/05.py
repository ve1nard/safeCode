import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use the tarfile directly
        tar.extractall("/tmp/unpack")

    # the files are now in /tmp/unpack, we can use the os module to
    # list them
    for f in os.listdir("/tmp/unpack"):
        print(f)
