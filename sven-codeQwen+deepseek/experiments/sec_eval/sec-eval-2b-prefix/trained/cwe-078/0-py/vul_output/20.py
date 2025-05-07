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
    ls = "ls -l {}".format(dirname)
    # call subprocess.call to run the command
    subprocess.call(ls, shell=True)
    # return a file
    return "ls -l {}".format(dirname)
