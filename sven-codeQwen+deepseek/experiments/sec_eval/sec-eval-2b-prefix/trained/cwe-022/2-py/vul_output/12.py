import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar.getmembers():
            if member.name.startswith("/"):
                tar.extract(member, "/tmp/unpack")

    # move the extracted files to the /home/user/unpack directory
    for f in os.listdir("/tmp/unpack"):
        if os.path.isdir(f):
            shutil.move(f, "/home/user/unpack")
        else:
            shutil.move(f, "/home/user/unpack/%s" % f)
