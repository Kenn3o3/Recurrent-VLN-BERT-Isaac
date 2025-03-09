from flask import Flask, render_template
from flask_socketio import SocketIO
import base64
import threading
import cv2
from queue import Queue

class WebUI:
    def __init__(self, action_queue=None):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, async_mode='threading')
        self.current_data = {}
        self.action_queue = action_queue  # Queue to store user actions
        self.setup_routes()
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    @staticmethod
    def image_to_base64(img_np):
        """Convert numpy image to base64 string"""
        _, buffer = cv2.imencode('.jpg', img_np)
        return base64.b64encode(buffer).decode('utf-8')

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html', **self.current_data)

        @self.socketio.on('connect')
        def handle_connect():
            self.socketio.emit('initial_data', self.current_data)

        @self.socketio.on('action')
        def handle_action(data):
            action = data['action']
            if self.action_queue is not None:
                self.action_queue.put(action)
            print(f"Received action: {action}")

    def update_data(self, data):
        self.current_data = data
        self.socketio.emit('update', data)

    def run(self):
        self.socketio.run(self.app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)