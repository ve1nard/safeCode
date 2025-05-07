import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for tarinfo in tar:
            # get the file name and the file path
            file_name = tarinfo.name.split("/")
            file_path = "/tmp/"
            # extract all the files
            for file_name_part in file_name:
                # join the path and the file name to get the final file path
                file_path += file_name_part + "/"
                # extract the file
                tar.extract(tarinfo, file_path)
