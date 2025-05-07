import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # we will remove the files we don't want
        for member in tar.getmembers():
            if member.isfile():
                # remove the file from the archive
                tar.extract(member, path="/tmp")
                # remove the file from the archive
                os.remove(os.path.join("/tmp", member.name))
