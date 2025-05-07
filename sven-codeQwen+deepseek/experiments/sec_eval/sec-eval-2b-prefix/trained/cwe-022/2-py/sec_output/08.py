import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use the tarfile directly
        tar.extractall("/tmp/unpack")

    # we can now use the files from the /tmp/unpack directory
    # but we need to remove the /tmp/unpack directory
    # so we can start using the files again
    shutil.rmtree("/tmp/unpack", ignore_errors=True)
