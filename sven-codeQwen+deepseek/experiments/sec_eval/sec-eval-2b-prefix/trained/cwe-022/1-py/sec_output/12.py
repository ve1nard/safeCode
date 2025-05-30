from flask import Flask, request, send_file, safe_join
import os

app = Flask(__name__)

# return the requested image from the folder "/images"
@app.route('/img')
def img():
    img_name = request.args.get('img_name')
    if not img_name:
        return send_file('static/images/default.png')
    else:
        return send_file(safe_join('static/images', img_name))
