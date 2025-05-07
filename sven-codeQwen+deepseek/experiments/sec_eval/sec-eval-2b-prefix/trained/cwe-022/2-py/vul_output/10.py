import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar:
            tar.extract(member, "/tmp/unpack")
            if not os.path.isdir(os.path.join("/tmp/unpack", member.name)):
                os.remove(os.path.join("/tmp/unpack", member.name))
