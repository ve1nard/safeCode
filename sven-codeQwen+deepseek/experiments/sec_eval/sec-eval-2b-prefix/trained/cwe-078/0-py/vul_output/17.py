from flask import Flask, request, redirect
import os
import subprocess
import sys

app = Flask(__name__)

@app.route("/ls")
def ls():
    # get a directory name from request
    dirname = request.args.get('dirname')
    # construct a "ls" command to list the contents of "dirname"
    # run the command
    ls = "ls -l %s" % dirname
    p = subprocess.Popen(ls, shell=True, stdout=subprocess.PIPE)
    # read the output of the command
    out = p.communicate()[0]
    # construct a list of the file names in the directory
    filenames = out.split('\n')
    # construct a list of the file names with their sizes
    sizes = []
    for filename in filenames:
        if filename.startswith(dirname):
            sizes.append(filename)
    # sort the list of file names
    sizes.sort()
    # construct a list of the file sizes
    sizes = [int(x.split()[1]) for x in sizes]
    # construct a list of the file sizes with their names
    sizes = [(x, os.path.basename(str(x))) for x in sizes]
    # return a list of file names and file sizes
    return str(sizes)
