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
    # run the command and capture its output
    cmd = ["ls", "--color=never", dirname]
    ls = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # read the output of the command
    ls_out, ls_err = ls.communicate()
    # print the output of the command
    print ls_out
    # return the output of the command
    return ls_out
