import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for tarinfo in tar:
            # we don't care about the tarinfo.type
            # we only care about the tarinfo.name
            if tarinfo.name == "":
                continue
            if tarinfo.isreg():
                # we don't care about the file contents,
                # we only care about the file name
                # so we can just extract the file to /tmp/unpack
                with tar.extractfile(tarinfo) as f:
                    with open("/tmp/unpack/" + tarinfo.name, "wb") as f2:
                        f2.write(f.read())
            else:
                # we don't care about the file contents,
                # we only care about the file name
                # so we can just extract the file to /tmp/unpack
                tar.extract(tarinfo, "/tmp/unpack")
