import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        for member in tar:
            # member is a TarInfo object
            # member.name is the file or directory name inside the tar file
            # member.path is the absolute path to the file or directory
            # member.type is the file type (file, dir, etc.)
            # member.mode is the file mode (r, w, etc.)
            if member.isdir():
                os.makedirs(member.path, exist_ok=True)
            else:
                # member.path is the absolute path to the file inside the tar file
                # member.name is the file name inside the tar file
                # member.type is the file type (file, dir, etc.)
                # member.mode is the file mode (r, w, etc.)
                # member.size is the file size in bytes
                # member.date_time is the file modification time
                # member.pax_headers are the pax headers
                with open(member.path, "wb") as f:
                    # open the file with write mode
                    # write the file content to the file
                    f.write(tar.extractfile(member).read())
