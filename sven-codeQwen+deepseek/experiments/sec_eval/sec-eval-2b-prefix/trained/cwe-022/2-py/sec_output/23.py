import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just use the tarfile directly
        for tarinfo in tar:
            # we only extract the files that are not directories
            if tarinfo.isdir():
                continue
            # we can just use the tarfile directly
            tar.extract(tarinfo, path=os.getcwd())
