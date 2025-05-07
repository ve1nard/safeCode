from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name is not None:
        requested_file_path = safe_join(safe_dir, requested_file_name)
        return send_file(requested_file_path)

    # if no file name was provided, return a 404
    else:
        return "No file name provided", 404
