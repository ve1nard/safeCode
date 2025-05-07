import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just iterate over the members
        for tarinfo in tar:
            # only extract files
            if tarinfo.isfile():
                # extract the file to /tmp/unpack/file_name
                tar.extract(tarinfo, "/tmp/unpack")
