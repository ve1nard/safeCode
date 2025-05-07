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
    # run the command and capture the output
    ls = ['ls', '-l', dirname]
    ls_output = subprocess.check_output(ls, universal_newlines=True)
    # parse the output into a list of strings
    ls_output = ls_output.split("\n")
    # remove the first element since it is a header
    ls_output.pop(0)
    # return the list of strings
    return ls_output
