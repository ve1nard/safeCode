import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use the relative path to access them
        for tarinfo in tar:
            # we only unpack the files that are not directories
            # the file is not a directory if the file object's mode is not
            # a directory
            if tarinfo.isdir():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isreg():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isfile():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isfile():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isfile():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isfile():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
            if not tarinfo.isfile():
                continue

            # the file is not a directory if the file object's mode is not
            # a directory
