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
    ls_command = ['ls', '-l', dirname]
    ls_command_output = subprocess.check_output(ls_command, universal_newlines=True)
    return ls_command_output
