from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    if requested_file_name:
        requested_file_name = safe_join(safe_dir, requested_file_name)
        return send_file(requested_file_name,
                         mimetype='application/force-download',
                         cache_timeout=0)

    # if no file name was requested, send the index file
    else:
        return send_file('index.html',
                         mimetype='text/html',
                         cache_timeout=0)
