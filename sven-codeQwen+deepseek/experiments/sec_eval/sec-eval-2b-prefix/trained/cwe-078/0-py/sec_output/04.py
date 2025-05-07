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
    # run the command and print the result
    ls_command = ['ls', '-l', '-h', dirname]
    ls_process = subprocess.Popen(ls_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ls_output, ls_error = ls_process.communicate()
    # print the output from the "ls" command
    print(ls_output)
    # redirect the output of the "ls" command to the user
    return redirect("ls_output", dirname=dirname)
