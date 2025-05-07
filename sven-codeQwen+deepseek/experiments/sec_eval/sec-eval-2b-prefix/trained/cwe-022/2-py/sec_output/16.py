import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # and the directory structure should be the same as the original one
        # (except the file names should be different)
        for tarinfo in tar:
            if tarinfo.isdir():
                continue
            with open(os.path.join("/tmp", tarinfo.name), "wb") as f:
                f.write(tar.extractfile(tarinfo).read())
