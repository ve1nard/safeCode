import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        tar.extractall("/tmp/unpack")

    # now we can work with the files
    with open("/tmp/unpack/README.rst") as f:
        readme = f.read()

    with open("/tmp/unpack/requirements.txt") as f:
        requirements = f.read()

    return readme, requirements
