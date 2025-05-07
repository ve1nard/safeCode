from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    file_path = safe_join(safe_dir, requested_file_name)

    # if the file does not exist, return a 404 error
    if not os.path.isfile(file_path):
        return '404', 404

    # read the file
    with open(file_path, 'r') as f:
        file_contents = f.read()

    # return the file contents
    return file_contents, 200, {'Content-Type': 'text/plain'}
