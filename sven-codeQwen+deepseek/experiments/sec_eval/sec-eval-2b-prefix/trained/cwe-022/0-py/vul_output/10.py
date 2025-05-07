from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    requested_file_path = os.path.join(safe_dir, requested_file_name)
    if os.path.isfile(requested_file_path):
        with open(requested_file_path) as file:
            return send_file(requested_file_path)

    # if the requested file is not found, send a 404 error
    return "File not found"
