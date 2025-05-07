from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name:
        safe_file = safe_join(safe_dir, requested_file_name)
        if os.path.isfile(safe_file):
            return send_file(safe_file)

    # if the file name is not provided, return an error
    return 'No file was provided', 400
