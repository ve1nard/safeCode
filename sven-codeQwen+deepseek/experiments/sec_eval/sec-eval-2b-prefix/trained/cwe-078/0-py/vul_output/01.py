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
    cmd = "ls -l {}".format(dirname)
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output = p.communicate()[0]
    # return the contents of the file
    return output
