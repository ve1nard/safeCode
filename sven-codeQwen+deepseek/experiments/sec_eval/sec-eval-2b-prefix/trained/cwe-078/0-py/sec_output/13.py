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
    ls_cmd = ['ls', '-lh', dirname]
    print("ls command: " + str(ls_cmd))
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    # read the output of the command
    ls_output = ls_proc.communicate()[0]
    # print the output to the user
    print(ls_output)
    # return the output to the user
    return redirect("/")
