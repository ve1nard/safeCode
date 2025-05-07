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
    cmd = "ls -l {}".format(dirname)
    # print the command
    print cmd
    # run the command
    subprocess.call(cmd, shell=True)
    # return a "200 OK"
    return "200 OK"
