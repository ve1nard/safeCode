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
    ls_cmd = "ls -l " + dirname
    ls_proc = subprocess.Popen(ls_cmd, shell=True, stdout=subprocess.PIPE)
    ls_out = ls_proc.communicate()[0]
    # construct a list of file names
    ls_list = ls_out.split('\n')
    # remove the first line (header)
    ls_list = ls_list[1:]
    # construct a list of file names
    ls_file_list = []
    for entry in ls_list:
        if entry!= '.' and entry!= '..':
            ls_file_list.append(entry)
    return render_template('ls.html', file_list=ls_file_list)
