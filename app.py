# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from camera import ObjectDetection

app = Flask(__name__)
@app.route("/")
def main():
    return render_template("index.html")

def gen(camera):
    while True:
        frame = camera.main()
        if frame != "":
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    id = 0
    return Response(gen(ObjectDetection(id)), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Serve the app with gevent
    app.run(host='127.0.0.1', threaded=True, debug = True)
