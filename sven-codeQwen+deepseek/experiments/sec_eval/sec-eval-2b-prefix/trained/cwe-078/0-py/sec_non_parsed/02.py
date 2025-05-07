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
    # run the command and print the output to the terminal
    ls_cmd = ["/bin/ls", "-l", dirname]
    ls_proc = subprocess.Popen(ls_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # read the output of the command and print it to the terminal
    for line in ls_proc.stdout:
        print line,
    # close the "ls" command's stdout pipe
    ls_proc.stdout.close()
    # close the "ls" command's stderr pipe
    ls_proc.stderr.close()
    # return the "ls" command's exit code
    return str(ls_proc.wait())
