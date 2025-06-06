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
    ls_cmd = "ls -l {}".format(dirname)
    ls_proc = subprocess.Popen(ls_cmd, shell=True, stdout=subprocess.PIPE)
    ls_output = ls_proc.communicate()[0]
    # return the output of the command
    return ls_output
