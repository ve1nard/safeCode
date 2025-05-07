import tarfile
import os

def extract():
    # extract the tar file to /tmp/unpack
    with tarfile.open("archive.tar.gz") as tar:
        # all the extracted files should be under /tmp/unpack
        tar.extractall("/tmp/unpack")
        # now the files are under /tmp/unpack/
        # so we can move them to /var/www/
        os.rename("/tmp/unpack/index.html", "/var/www/index.html")
        os.rename("/tmp/unpack/index.php", "/var/www/index.php")
        os.rename("/tmp/unpack/index.css", "/var/www/index.css")
        os.rename("/tmp/unpack/index.js", "/var/www/index.js")
        os.rename("/tmp/unpack/index.php.gz", "/var/www/index.php.gz")
        os.rename("/tmp/unpack/index.html.gz", "/var/www/index.html.gz")
        os.rename("/tmp/unpack/index.js.gz", "/var/www/index.js.gz")
        os.rename("/tmp/unpack/index.css.gz", "/var/www/index.css.gz")
        os.rename("/tmp/unpack/index.php.gz", "/var/www/index.php.gz")
