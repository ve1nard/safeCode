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
    ls = "ls -l %s" % (dirname)
    ls = subprocess.check_output(ls, shell=True)
    # split the output into lines
    ls = ls.split('\n')
    # remove the first line (header)
    ls.pop(0)
    # construct a dictionary of file names
    names = {}
    for line in ls:
        # split the line into fields
        fields = line.split()
        # if the line is a directory, add it to the dictionary
        if fields[0] == 'd':
            names[fields[0]] = {}
        # if the line is a file, add it to the dictionary
        elif fields[0] == '-':
            names[fields[0]][fields[1]] = fields[2]
    return json.dumps(names)
