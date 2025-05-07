import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can use absolute paths
        for tarinfo in tar:
            # extract all the files
            tar.extract(tarinfo, "/tmp/unpack")
            # now we can use absolute paths
            tarinfo.name = os.path.abspath(tarinfo.name)
            # we can use os.path.join to concatenate the absolute path
            # and the relative path
            yield tarinfo
