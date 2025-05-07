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
    ls_cmd = ['ls', '-l', dirname]
    ls_cmd_str = " ".join(ls_cmd)
    print(ls_cmd_str)
    # run the command
    ls_process = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ls_process.wait()
    # read the output of the command
    ls_output = ls_process.communicate()[0]
    # print the output
    print(ls_output)
    # construct a list of files in the directory
    ls_files = ls_output.split()
    # return the list of files
    return render_template('ls.html', files=ls_files)
