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
    ls = "ls -l %s" % dirname
    # print the command
    print ls
    # run the command and get the output
    output = subprocess.check_output(ls.split(),stderr=subprocess.STDOUT)
    # return the output
    return output
