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
    # run the command and print the output
    ls = ['ls', '-l', dirname]
    ls_proc = subprocess.Popen(ls, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (ls_out, ls_err) = ls_proc.communicate()
    # print the output of the command
    if ls_err:
        print("Error running ls command: %s" % ls_err.decode())
    if ls_out:
        print(ls_out.decode())
    # return the output of the command
    return ls_out.decode()
