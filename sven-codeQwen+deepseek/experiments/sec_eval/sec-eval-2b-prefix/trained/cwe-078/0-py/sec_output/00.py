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
    ls_cmd = ["ls", "-l", dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # read the output of the command
    ls_proc_stdout, ls_proc_stderr = ls_proc.communicate()
    # parse the output to get the list of files
    ls_files = ls_proc_stdout.split()
    # return the list of files
    return json.dumps(ls_files)
