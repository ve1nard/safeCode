from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    file_path = os.path.join(safe_dir, requested_file_name)

    if not os.path.isfile(file_path):
        return '404 - File not found'

    with open(file_path, 'rb') as f:
        return f.read()

    return '200 - OK'
