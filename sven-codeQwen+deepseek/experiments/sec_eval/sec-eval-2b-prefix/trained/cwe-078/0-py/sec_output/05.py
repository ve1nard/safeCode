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
    # run the command and get the output
    ls_cmd = ['ls', '-l', dirname]
    ls_output = subprocess.check_output(ls_cmd, stderr=subprocess.STDOUT)
    # split the output into lines, and remove the last line which is the empty string
    ls_lines = ls_output.splitlines()[:-1]
    # create a dictionary of the output, where the keys are the file names and the values are the file sizes
    ls_dict = {}
    # loop over the lines, and build a dictionary of file names and file sizes
    for line in ls_lines:
        # split the line into file name and file size
        file_name, file_size = line.split()
        # add the file name and file size to the dictionary
        ls_dict[file_name] = file_size
    # return the dictionary of file names and file sizes
    return json.dumps(ls_dict)
