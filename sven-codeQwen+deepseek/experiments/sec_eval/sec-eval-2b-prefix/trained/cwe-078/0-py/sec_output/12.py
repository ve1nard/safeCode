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
    ls_cmd = ['ls', '-l', '-h', dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # read the output of the command
    ls_out, ls_err = ls_proc.communicate()
    # return the output of the command
    return redirect('/')
