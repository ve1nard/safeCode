import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just use the same code as for the zip file
        for tarinfo in tar:
            # we only unpack.py files
            if tarinfo.isreg() and tarinfo.name.endswith('.py'):
                # we need to change the directory to the one in the archive
                current_dir = os.getcwd()
                os.chdir(os.path.dirname(tarinfo.name))
                # we can unpack the file
                tar.extract(tarinfo, os.getcwd())
                # and change back to the original directory
                os.chdir(current_dir)
