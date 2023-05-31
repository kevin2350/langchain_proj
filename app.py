from flask import Flask, render_template, request, flash

app = Flask(__name__)

@app.route("/default")
def index():
    return render_template("/frontend/index.html")