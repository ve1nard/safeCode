from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    requested_file = safe_join(safe_dir, requested_file_name)
    return send_file(requested_file, mimetype='application/octet-stream')
