from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

@app.route("/read")
def read():
    requested_file_name = request.args.get('filename')
    safe_dir = '/safe/'

    # read the requested file from the safe directory
    # if the file does not exist, send a 404
    if requested_file_name:
        safe_file = safe_join(safe_dir, requested_file_name)
        if not os.path.exists(safe_file):
            return send_file(safe_file)

    # if the file does exist, send it to the client
    else:
        return send_file(safe_join(safe_dir, 'index.html'))
