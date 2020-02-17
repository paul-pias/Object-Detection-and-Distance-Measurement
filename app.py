# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from camera import Camera
from object_detection import object_detection

app = Flask(__name__)
camera = Camera(0)
@app.route("/")
def main():
    return render_template("test.html")

def gen(camera):
    while True:
        frame,l,feedback = camera.get_frame()
        if frame != "":
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():

    return Response(gen(camera), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('localhost', 5000), app)
    http_server.serve_forever()