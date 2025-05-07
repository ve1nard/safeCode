import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use absolute paths
        for tarinfo in tar:
            # get the absolute path of the file
            # and then use it as the destination
            destination = os.path.abspath(os.path.join("/tmp/unpack", tarinfo.name))
            # if the destination exists, remove it
            if os.path.exists(destination):
                os.remove(destination)
            # extract the file
            tar.extract(tarinfo, "/tmp/unpack")
