from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name is None:
        return 'No file specified', 400

    # check the file exists
    if not os.path.exists(safe_join(safe_dir, requested_file_name)):
        return 'File not found', 404

    # send the file
    return send_file(safe_join(safe_dir, requested_file_name))
