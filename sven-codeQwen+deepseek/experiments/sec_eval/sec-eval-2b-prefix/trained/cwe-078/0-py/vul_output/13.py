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
        "ls -l {}".format(dirname), shell=True)
    # split the ls output into lines
    ls = ls.splitlines()
    # return a list of lines
    return ls
