# -*- coding: utf-8 -*-
"""
    :author: Grey Li <withlihui@gmail.com>
    :copyright: (c) 2017 by Grey Li.
    :license: MIT, see LICENSE for more details.
"""
import os
import hashlib
import random

from flask import Flask, render_template, request
from flask_dropzone import Dropzone

basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(basedir, 'uploads'), exist_ok=True)

app = Flask(__name__)

app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static', 'uploads'),
    DONE_PATH=os.path.join(basedir, 'static', 'done'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=5,
    DROPZONE_MAX_FILES=1,
)

dropzone = Dropzone(app)


@app.route('/', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()
        _, ext = os.path.splitext(f.filename)
        fp = open(os.path.join(app.config['UPLOADED_PATH'], "%s%s" % (sha256, ext)), 'wb')
        fp.write(data)
        fp.close()
        return sha256
    return render_template('index.html')


@app.route('/jobs', methods=['POST', 'GET'])
def jobs():
    if request.method == 'POST':
        f = request.files.get('file')
        f.save(os.path.join(app.config['DONE_PATH'], f.filename))  # sha256.jpg
        return f.filename
    done = list(map(lambda x: x.split('.')[0], os.listdir(app.config['DONE_PATH'])))
    jobs = list(filter(lambda x: x.split('.')[0] not in done, os.listdir(app.config['UPLOADED_PATH'])))
    if not jobs:
        return "empty"
    return random.choice(jobs)


if __name__ == '__main__':
    app.run(debug=True)
