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
    cmd = ['ls', '-l', dirname]
    print("ls: ", " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, errors = p.communicate()
    # output is a list of lines, one for each file in the directory
    # each line is a list of the file's name, size, and modification date
    # print the output
    print(output.decode('utf-8'))
    return output.decode('utf-8')
