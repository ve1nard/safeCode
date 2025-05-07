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
    ls = subprocess.check_output(
        "ls -l %s" % dirname, shell=True)
    # convert the output into a list of files and directories
    ls = ls.split('\n')
    # remove the first line (it's a header)
    ls = ls[1:]
    # return the list of files and directories
    return '<ul>%s</ul>' % '\n'.join(
        [os.path.join(dirname, f) for f in ls])
