from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name:
        file_path = safe_join(safe_dir, requested_file_name)
        if os.path.isfile(file_path):
            return send_file(file_path)
        else:
            return "File not found", 404

    # return a list of all files in the safe directory
    else:
        return send_file_list(safe_dir)
