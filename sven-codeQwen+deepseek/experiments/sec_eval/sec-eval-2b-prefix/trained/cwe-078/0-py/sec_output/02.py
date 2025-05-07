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
    # run the command and print the output to the browser
    ls_cmd = ['ls', '-l', dirname]
    p = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = p.communicate()
    return output
