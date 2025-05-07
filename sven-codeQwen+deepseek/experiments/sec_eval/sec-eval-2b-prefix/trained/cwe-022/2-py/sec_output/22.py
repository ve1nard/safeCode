import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        # so we can just use the tarfile directly
        # we can also filter the files we want to extract
        # tar.extractall() will return a list of all the extracted files
        # if we want to extract only the files we want
        # tar.extractall(members=["file1.txt", "file2.txt"])
        # will extract only the files we want
        # tar.extractall(members=["file1.txt", "file2.txt"])
        # will extract only the files we want
        tar.extractall(path="/tmp/unpack")
        # we can also extract files by using the member attribute
        # tar.extractall(members=tar.getmembers())
        # will extract all the files in the archive
        # tar.extractall(members=tar.getmembers())
        # will extract all the files in the archive
