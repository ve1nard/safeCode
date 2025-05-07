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
    # run the command, and print the output
    ls_cmd = ['ls', '-l', dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ls_proc.communicate()
    # parse the output and return the list of file names
    ls_output = ls_proc.communicate()[0].splitlines()
    ls_output = [x.strip() for x in ls_output]
    # return the list of file names
    return json.dumps(ls_output)
