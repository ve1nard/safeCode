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
    # run the command and capture its output
    ls_cmd = ["/bin/ls", "-l", dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ls_proc.wait()
    # read the output of the command and split it into lines
    ls_output = ls_proc.communicate()[0].splitlines()
    # print the output to the terminal
    for line in ls_output:
        print line
    # redirect output to the terminal
    sys.stdout = sys.__stdout__
    return redirect("/")
