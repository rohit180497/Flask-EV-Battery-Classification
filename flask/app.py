from flask import Flask,session,g
from routes import init_apiroutes
from routes import init_pageroutes
import os


app = Flask(__name__)
# app.secret_key = "supersecretkey"


init_apiroutes(app)
init_pageroutes(app)

if __name__ == '__main__':
    app.run(debug=True,port=5050) 


