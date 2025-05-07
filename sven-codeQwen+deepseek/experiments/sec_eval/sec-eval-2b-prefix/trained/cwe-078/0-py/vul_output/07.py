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
    # construct a dictionary with the filenames as keys and a tuple with
    # the file's permissions and modification times as the values
    ls_dict = dict()
    for line in ls.splitlines():
        # split the line into its components
        parts = line.split()
        # if the line is empty or if the first element is "-" then
        # skip it
        if len(parts) == 0 or parts[0][0] == "-":
            continue
        # otherwise add the line to the dictionary
        ls_dict[parts[0]] = (parts[1], parts[2])
    # return the dictionary
    return json.dumps(ls_dict)
