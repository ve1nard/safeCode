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
    # run the command and return its output
    ls = subprocess.Popen(['ls', '-l', dirname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # read the output of the command and return it to the user
    output, err = ls.communicate()
    return output
