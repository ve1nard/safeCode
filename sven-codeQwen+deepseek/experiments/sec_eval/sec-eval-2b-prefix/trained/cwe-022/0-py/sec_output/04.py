from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    file_path = safe_join(safe_dir, requested_file_name)
    file_data = safe_open(file_path, 'rb').read()

    # send the file to the client
    return send_file(file_data,
                     mimetype='application/force-download',
                     mimetype_params={'filename': requested_file_name})
