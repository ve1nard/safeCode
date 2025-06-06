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
    ls = "ls -l " + dirname
    ls_output = subprocess.check_output(ls, shell=True)
    # construct a dictionary from the output of the command
    ls_output_dict = {}
    for line in ls_output.split("\n"):
        # split the line into key, value
        key, value = line.split(" ")
        # add the key, value pair to the dictionary
        ls_output_dict[key] = value
    return ls_output_dict
