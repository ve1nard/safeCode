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
    ls_cmd = ['ls', '-l', dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # get the output of the command
    ls_output = ls_proc.communicate()[0]
    # return the output
    return ls_output
