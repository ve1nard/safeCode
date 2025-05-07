import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just use the tarfile.extractall() method
        tar.extractall("/tmp/unpack")

    # now we can access all the files in /tmp/unpack
    # we can use os.path.join() to compose the absolute path
    # for example: os.path.join("/tmp/unpack", "file.txt")
    # will compose the absolute path to /tmp/unpack/file.txt

    # we can then use the os.path.exists() function to check if the file exists
    # for example: os.path.exists("/tmp/unpack/file.txt")
    # will return true if the file /tmp/unpack/file.txt exists
    # and false if it does not exist
    # this is the same as: if os.path.exists("/tmp/unpack/file.txt")
    # but more explicit

    # we can also use the os.path.isdir() function to check if a directory exists
    # for example: os.path.isdir("/tmp/unpack/")
    # will return true if the directory /tmp/unpack/ exists
    # and false if it does not exist
