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
    # run the command and return the output
    ls_cmd = ['ls', '-l', '-1', dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ls_out, ls_err = ls_proc.communicate()
    # parse the output and return a list of file names
    ls_output = ls_out.splitlines()
    return ls_output
