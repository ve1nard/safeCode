import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar:
            # if the file is a directory, then extract it
            if member.isfile():
                # extract the file
                tar.extract(member, "/tmp/unpack")
    return "/tmp/unpack"
