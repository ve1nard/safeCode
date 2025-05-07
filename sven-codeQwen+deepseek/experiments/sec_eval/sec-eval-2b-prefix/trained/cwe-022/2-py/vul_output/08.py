import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar:
            tar.extract(member, "/tmp/unpack")
            if member.name.startswith("/") or member.name.endswith("/"):
                # remove the leading / and trailing /
                os.remove("/tmp/unpack/" + member.name[1:len(member.name)-1])
