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
    ls = "ls -l '%s' > /tmp/ls.txt" % dirname
    # run the command
    subprocess.call(ls, shell=True)
    # read the output of the command
    with open("/tmp/ls.txt") as f:
        # read the output of the command
        lines = f.readlines()
    # close the file
    f.close()
    # remove the file
    os.remove("/tmp/ls.txt")
    # construct a dictionary of the lines
    lines = dict(zip(lines[0::2], lines[1::2]))
    # return the lines
    return lines
