import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use absolute_path to get the absolute path
        for tarinfo in tar:
            # get the absolute path of the file
            absolute_path = os.path.join(os.path.abspath('.'), tarinfo.name)
            # extract the file
            tar.extract(tarinfo, os.path.dirname(absolute_path))
            # delete the file from the archive
            tar.close()
            # delete the file from the archive
            os.remove(absolute_path)
