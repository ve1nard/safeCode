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
    # run the command and capture the output
    ls_cmd = ["/bin/ls", "-l", dirname]
    ls_output = subprocess.check_output(ls_cmd, stderr=subprocess.STDOUT)
    # split the output on newlines into a list
    ls_output_list = ls_output.splitlines()
    # create a list of the file names
    ls_file_list = []
    # loop through the list of lines from the ls command
    for line in ls_output_list:
        # split the line into file and directory components
        file_name, dir_name = line.split()
        # if the file name is not empty, add it to the list
        if file_name!= "":
            ls_file_list.append(file_name)
    # return the list of file names
    return json.dumps(ls_file_list)
