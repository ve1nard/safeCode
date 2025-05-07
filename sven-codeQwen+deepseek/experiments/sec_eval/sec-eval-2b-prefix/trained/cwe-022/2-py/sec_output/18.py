import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for tarinfo in tar:
            if tarinfo.isreg():
                # the file is a regular file
                # we extract it to /tmp/unpack/file_name
                with open(os.path.join("/tmp/unpack", tarinfo.name), "wb") as f:
                    f.write(tar.extractfile(tarinfo).read())
