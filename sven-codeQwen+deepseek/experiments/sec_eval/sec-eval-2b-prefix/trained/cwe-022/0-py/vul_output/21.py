from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if os.path.isfile(safe_dir + requested_file_name):
        file = open(safe_dir + requested_file_name)
        return send_file(file)

    # if the requested file is not found, return a 404 error
    return '<h1>File not found</h1>'
