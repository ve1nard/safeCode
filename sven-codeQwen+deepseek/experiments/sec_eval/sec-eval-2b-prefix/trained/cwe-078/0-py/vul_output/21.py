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
    ls_lines = ls.splitlines()
    # construct a list of lines to return
    lines = []
    for line in ls_lines:
        # if the line is a directory, add a slash to it
        if line.startswith('drw'):
            line = line + '/'
        # add the line to the list
        lines.append(line)
    # return the list of lines
    return lines
