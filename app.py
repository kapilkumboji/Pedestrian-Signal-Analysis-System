from flask import Flask, render_template, jsonify, request
import threading

app = Flask(__name__)

# Shared data for pedestrian count, vehicle count, and red light duration
shared_data = {
    "pedestrian_count": 0,
    "vehicle_count": 0,
    "red_light_duration": 10
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data', methods=['GET'])
def data():
    return jsonify(shared_data)

@app.route('/increase_duration', methods=['POST'])
def increase_duration():
    shared_data["red_light_duration"] += 10
    return jsonify(success=True)

def run_flask():
    app.run(debug=True, use_reloader=False)

# Start Flask server in a separate thread
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()