from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    requested_file_path = os.path.join(safe_dir, requested_file_name)
    with open(requested_file_path) as f:
        return f.read()

    # return an error if the file was not found
    return "File not found"
