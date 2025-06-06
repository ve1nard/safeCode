import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar.getmembers():
            tar.extract(member, "/tmp/unpack")
            if member.name.endswith(".pyc"):
                os.unlink(member.name)
