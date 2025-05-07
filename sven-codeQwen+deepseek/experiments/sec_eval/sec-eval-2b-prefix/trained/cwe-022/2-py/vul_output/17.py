import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar:
            # extract the file
            tar.extract(member, "/tmp/unpack")
            # get the file name
            file_name = member.name
            # remove the '/' char at the end of the file name
            if file_name[-1] == '/':
                file_name = file_name[0:-1]
            # move the file to /home/user/Desktop/
            os.rename(os.path.join("/tmp/unpack", file_name),
                      os.path.join(os.getcwd(), file_name))
