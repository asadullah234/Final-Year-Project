from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from simulation import run_simulation
import shutil
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from sqlalchemy import inspect, text
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import calendar

from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
from simulation import run_simulation
import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from sqlalchemy import inspect, text
import json
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import calendar
import io
from werkzeug.datastructures import FileStorage

# Initialize Flask application
app = Flask(__name__, static_url_path='/static')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:asad@localhost:5432/Traffic-Congestion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SIGNAL_STATE_FILE'] = 'signal_states.json'

# Initialize database
db = SQLAlchemy(app)

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLO model
model = YOLO("best.pt")

# Rest of your code...
from flask import Flask, render_template
from simulation import run_simulation

app = Flask(__name__)

@app.route('/simulation')
def simulation():
    result = run_simulation()  # this is what "runs" simulation.py
    return render_template('simulation.html', result=result)
app = Flask(__name__, static_url_path='/static')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:asad@localhost:5432/Traffic-Congestion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SIGNAL_STATE_FILE'] = 'signal_states.json'

db = SQLAlchemy(app)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
model = YOLO("best.pt")

class TrafficData(db.Model):
    __tablename__ = 'traffic_data'
    id = db.Column(db.Integer, primary_key=True)
    junction_id = db.Column(db.Integer, nullable=False)
    original_image_path = db.Column(db.String, nullable=False)
    processed_image_path = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    vehicle_counts = db.Column(db.JSON)
    total_vehicles = db.Column(db.Integer)
    congestion_level = db.Column(db.String)
    congestion_percentage = db.Column(db.Float)
    signal_timings = db.Column(db.JSON)
    analysis_completed = db.Column(db.Boolean, default=False)

#table for prediction data
class PredictionData(db.Model):
    __tablename__ = 'prediction_data'
    id = db.Column(db.Integer, primary_key=True)
    junction_id = db.Column(db.Integer, nullable=False)
    prediction_type = db.Column(db.String(20), nullable=False)  # 'hourly', 'daily', etc.
    prediction_date = db.Column(db.DateTime, nullable=False)
    congestion_prediction = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
  







def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def ensure_columns_exist():
    with app.app_context():
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns('traffic_data')]
        
        required_columns = {
            'congestion_level': 'ALTER TABLE traffic_data ADD COLUMN congestion_level VARCHAR',
            'congestion_percentage': 'ALTER TABLE traffic_data ADD COLUMN congestion_percentage FLOAT',
            'signal_timings': 'ALTER TABLE traffic_data ADD COLUMN signal_timings JSONB',
            'analysis_completed': 'ALTER TABLE traffic_data ADD COLUMN analysis_completed BOOLEAN DEFAULT FALSE'
        }
        
        for col_name, sql in required_columns.items():
            if col_name not in columns:
                try:
                    db.session.execute(text(sql))
                    db.session.commit()
                    print(f"Added column {col_name} to traffic_data table")
                except Exception as e:
                    db.session.rollback()
                    print(f"Error adding column {col_name}: {e}")

def get_signal_states():
    if not os.path.exists(app.config['SIGNAL_STATE_FILE']):
        initial_state = {
            'active_junction': None,
            'signal_states': {
                1: {'state': 'red', 'remaining': 0, 'last_update': None},
                2: {'state': 'red', 'remaining': 0, 'last_update': None},
                3: {'state': 'red', 'remaining': 0, 'last_update': None},
                4: {'state': 'red', 'remaining': 0, 'last_update': None}
            }
        }
        with open(app.config['SIGNAL_STATE_FILE'], 'w') as f:
            json.dump(initial_state, f)
        return initial_state
    
    with open(app.config['SIGNAL_STATE_FILE'], 'r') as f:
        return json.load(f)

def update_signal_states(new_states):
    with open(app.config['SIGNAL_STATE_FILE'], 'w') as f:
        json.dump(new_states, f)

def analyze_image(image_path, junction_id):
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Failed to load image"}
    
    results = model(img)
    vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bike', 'rickshaw', 'pickup']
    counts = {v: 0 for v in vehicle_classes}
    total = 0
    
    processed_img = img.copy()
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0].item()
            class_name = model.names[class_id].lower()
            
            if class_name in counts:
                counts[class_name] += 1
                total += 1
            
            color = (0, 255, 0)
            cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(processed_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    processed_filename = f"processed_junction_{junction_id}_{int(time.time())}.jpg"
    processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, processed_img)
    
    congestion_pct = min((total / 50) * 100, 100)
    if congestion_pct < 30:
        level = "Low"
    elif congestion_pct < 70:
        level = "Medium"
    else:
        level = "High"
    
    # Calculate signal timings based on congestion
    base_green = 30  # Base green time in seconds
    adjusted_green = min(60, max(10, base_green + (congestion_pct - 50) * 0.6))
    
    return {
        "counts": counts,
        "total": total,
        "congestion_level": level,
        "congestion_pct": congestion_pct,
        "signal_timings": {
            "green": int(adjusted_green),
            "yellow": 3,
            "red": max(10, 120 - int(adjusted_green))  # Total cycle time of 120s
        },
        "processed_image": processed_filename,
        "original_image": os.path.basename(image_path)
    }
def generate_enhanced_predictions(junction_id):
    # Get historical data
    records = TrafficData.query.filter_by(junction_id=junction_id).order_by(TrafficData.created_at).all()
    
    if len(records) < 10:  # Need at least 10 data points
        return None
    
    # Prepare more detailed features
    data = []
    for record in records:
        timestamp = record.created_at
        data.append({
            'timestamp': timestamp.timestamp(),
            'hour': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'day_of_month': timestamp.day,
            'month': timestamp.month,
            'congestion': record.congestion_percentage
        })
    
    df = pd.DataFrame(data)
    
    # Features and target
    X = df[['hour', 'day_of_week', 'day_of_month', 'month']]
    y = df['congestion']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use Random Forest for better predictions
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Generate predictions for different timeframes
    now = datetime.utcnow()
    predictions = {
        'hourly': [],
        'daily': [],
        'weekly': [],
        'monthly': []
    }
    
    # Hourly predictions for next 24 hours
    for i in range(24):
        future_time = now + timedelta(hours=i+1)
        features = [[future_time.hour, future_time.weekday(), future_time.day, future_time.month]]
        pred = model.predict(features)[0]
        predictions['hourly'].append({
            'time': future_time.strftime('%Y-%m-%d %H:%M'),
            'congestion': max(0, min(100, pred))
        })
    
    # Daily predictions for next 7 days
    for i in range(7):
        future_time = now + timedelta(days=i+1)
        avg_congestion = 0
        # Predict for each hour and average
        for h in [8, 12, 17]:  # Morning, noon, evening
            features = [[h, future_time.weekday(), future_time.day, future_time.month]]
            pred = model.predict(features)[0]
            avg_congestion += max(0, min(100, pred))
        avg_congestion /= 3
        predictions['daily'].append({
            'date': future_time.strftime('%Y-%m-%d'),
            'day_name': calendar.day_name[future_time.weekday()],
            'congestion': avg_congestion
        })
    
    # Weekly average for next 4 weeks
    for i in range(4):
        week_start = now + timedelta(weeks=i+1)
        week_congestion = []
        for d in range(7):  # For each day in the week
            day = week_start + timedelta(days=d)
            for h in [8, 12, 17]:  # Sample hours
                features = [[h, day.weekday(), day.day, day.month]]
                pred = model.predict(features)[0]
                week_congestion.append(max(0, min(100, pred)))
        avg_week = sum(week_congestion) / len(week_congestion)
        predictions['weekly'].append({
            'week_number': (week_start - now).days // 7,
            'start_date': week_start.strftime('%Y-%m-%d'),
            'congestion': avg_week
        })
    
    # Monthly average for next 3 months
    for i in range(3):
        month_start = now + timedelta(days=30*(i+1))
        month_congestion = []
        # Sample days in the month
        for d in [1, 7, 14, 21, 28]:
            day = month_start + timedelta(days=d)
            for h in [8, 12, 17]:
                features = [[h, day.weekday(), day.day, day.month]]
                pred = model.predict(features)[0]
                month_congestion.append(max(0, min(100, pred)))
        avg_month = sum(month_congestion) / len(month_congestion)
        predictions['monthly'].append({
            'month_name': calendar.month_name[month_start.month],
            'year': month_start.year,
            'congestion': avg_month
        })

    
    return predictions
@app.route('/get_enhanced_predictions/<int:junction_id>')
def get_enhanced_predictions(junction_id):
    predictions = generate_enhanced_predictions(junction_id)
    if predictions:
        return jsonify(predictions)
    return jsonify({"error": "Not enough data for predictions"}), 400

@app.route('/export_stats/<int:junction_id>')
def export_stats(junction_id):
    # Get data for the junction
    records = TrafficData.query.filter_by(junction_id=junction_id).order_by(TrafficData.created_at).all()
    
    if not records:
        return jsonify({"error": "No data available for this junction"}), 404
    
    # Prepare data for CSV
    data = []
    for record in records:
        data.append({
            "timestamp": record.created_at.isoformat(),
            "total_vehicles": record.total_vehicles,
            "congestion_level": record.congestion_level,
            "congestion_percentage": record.congestion_percentage,
            "signal_timings": json.dumps(record.signal_timings)
        })
    
    df = pd.DataFrame(data)
    
    # Create CSV in memory
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    
    # Return as downloadable file
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename=junction_{junction_id}_stats.csv"}
    )

def generate_future_predictions(junction_id):
    # Get historical data for the junction
    records = TrafficData.query.filter_by(junction_id=junction_id).order_by(TrafficData.created_at).all()
    
    if len(records) < 2:
        return None
    
    # Prepare data for prediction
    data = []
    for record in records:
        data.append({
            'timestamp': record.created_at.timestamp(),
            'congestion': record.congestion_percentage
        })
    
    df = pd.DataFrame(data)
    X = df['timestamp'].values.reshape(-1, 1)
    y = df['congestion'].values
    
    # Train simple linear regression model
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for next 7 days
    future_timestamps = []
    now = datetime.utcnow()
    for i in range(1, 8):
        future_time = now + timedelta(days=i)
        future_timestamps.append(future_time.timestamp())
    
    future_X = np.array(future_timestamps).reshape(-1, 1)
    future_y = model.predict(future_X)
    
    # Format predictions
    predictions = []
    for i in range(7):
        date = (now + timedelta(days=i+1)).strftime('%Y-%m-%d')
        predictions.append({
            'date': date,
            'congestion': max(0, min(100, future_y[i]))
        })
    
    return predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/realtime')
def realtime():
    return render_template('realtime_result.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/simulation')
def simulation():
    result = run_simulation()
    return render_template('simulation.html', result=result)
@app.route('/get_signal_states')
def get_current_signal_states():
    signal_states = get_signal_states()
    # Update remaining times based on last_update
    current_time = time.time()
    for junction, state in signal_states['signal_states'].items():
        if state['last_update']:
            elapsed = current_time - state['last_update']
            state['remaining'] = max(0, state['remaining'] - elapsed)
            state['last_update'] = current_time
    update_signal_states(signal_states)
    return jsonify(signal_states)




@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

@app.route('/map_view')
def map_view():
    return render_template('map_view.html')

@app.route('/user_management')
def user_management():
    return render_template('user_management.html')

@app.route('/support')
def support():
    return render_template('support.html')

@app.route('/logout')
def logout():
    return redirect(url_for('index'))

# Add this import at the top
from flask import Response

# Update the analyze_junction route to fix issues
# Add these imports at the top if not already present
import shutil
from werkzeug.utils import secure_filename

# Update the analyze_image function to ensure proper image processing
def analyze_image(image_path, junction_id):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to load image"}
        
        # Make a copy of the original image for processing
        processed_img = img.copy()
        
        # Perform object detection
        results = model(img)
        
        vehicle_classes = ['car', 'bus', 'truck', 'motorcycle', 'bike', 'rickshaw', 'pickup']
        counts = {v: 0 for v in vehicle_classes}
        total = 0
        
        # Draw bounding boxes and labels
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()
                class_name = model.names[class_id].lower()
                
                if class_name in counts:
                    counts[class_name] += 1
                    total += 1
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color
                cv2.rectangle(processed_img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name} {confidence:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(processed_img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                
                # Draw label text
                cv2.putText(processed_img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save processed image
        processed_filename = f"processed_junction_{junction_id}_{int(time.time())}.jpg"
        processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
        
        # Ensure the image is saved in RGB format
        cv2.imwrite(processed_path, cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        
        # Calculate congestion metrics
        congestion_pct = min((total / 50) * 100, 100)  # Assuming 50 vehicles = 100% congestion
        if congestion_pct < 30:
            level = "Low"
        elif congestion_pct < 70:
            level = "Medium"
        else:
            level = "High"
        
        # Calculate signal timings
        base_green = 30  # Base green time in seconds
        adjusted_green = min(60, max(10, base_green + (congestion_pct - 50) * 0.6))
        
        return {
            "counts": counts,
            "total": total,
            "congestion_level": level,
            "congestion_pct": congestion_pct,
            "signal_timings": {
                "green": int(adjusted_green),
                "yellow": 3,
                "red": max(10, 120 - int(adjusted_green))  # Total cycle time of 120s
            },
            "processed_image": processed_filename,
            "original_image": os.path.basename(image_path)
        }
        
    except Exception as e:
        return {"error": f"Image processing error: {str(e)}"}

# Update the analyze_junction endpoint
@app.route('/analyze_junction', methods=['POST'])
def analyze_junction():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
    
    file = request.files['file']
    junction_id = int(request.form.get('junction_id', 1))
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file type"}), 400
    
    try:
        filename = secure_filename(f"junction_{junction_id}_{int(time.time())}.jpg")
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)
        
        analysis_result = analyze_image(original_path, junction_id)
        
        if "error" in analysis_result:
            os.remove(original_path)
            return jsonify({"success": False, "error": analysis_result["error"]}), 400
        
        traffic_data = TrafficData(
            junction_id=junction_id,
            original_image_path=filename,
            processed_image_path=analysis_result["processed_image"],
            vehicle_counts=analysis_result["counts"],
            total_vehicles=analysis_result["total"],
            congestion_level=analysis_result["congestion_level"],
            congestion_percentage=analysis_result["congestion_pct"],
            signal_timings=analysis_result["signal_timings"],
            analysis_completed=True
        )
        
        db.session.add(traffic_data)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "junction_id": junction_id,
            "result": analysis_result
        })
        
    except Exception as e:
        db.session.rollback()
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/get_processed_image/<filename>')
def get_processed_image(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except FileNotFoundError:
        return jsonify({"error": "Image not found"}), 404

@app.route('/get_dashboard_data')
def get_dashboard_data():
    # Get the latest data for all four junctions
    junctions_data = {}
    for junction_id in range(1, 5):
        latest_data = TrafficData.query.filter_by(
            junction_id=junction_id,
            analysis_completed=True
        ).order_by(TrafficData.created_at.desc()).first()
        
        if latest_data:
            junctions_data[junction_id] = {
                "original_image": latest_data.original_image_path,
                "processed_image": latest_data.processed_image_path,
                "total_vehicles": latest_data.total_vehicles,
                "congestion_level": latest_data.congestion_level,
                "congestion_percentage": latest_data.congestion_percentage,
                "signal_timings": latest_data.signal_timings,
                "vehicle_counts": latest_data.vehicle_counts,
                "timestamp": latest_data.created_at.isoformat()
            }
    
    return jsonify(junctions_data)

@app.route('/get_historical_data')
def get_historical_data():
    # Get data for the last 24 hours by default
    hours = int(request.args.get('hours', 24))
    time_threshold = datetime.utcnow() - timedelta(hours=hours)
    
    # Query data for all junctions
    historical_data = {1: [], 2: [], 3: [], 4: []}
    
    for junction_id in range(1, 5):
        records = TrafficData.query.filter(
            TrafficData.junction_id == junction_id,
            TrafficData.created_at >= time_threshold,
            TrafficData.analysis_completed == True
        ).order_by(TrafficData.created_at).all()
        
        for record in records:
            historical_data[junction_id].append({
                "timestamp": record.created_at.isoformat(),
                "congestion_percentage": record.congestion_percentage,
                "total_vehicles": record.total_vehicles
            })
    
    return jsonify(historical_data)

    

@app.route('/get_future_predictions/<int:junction_id>')
def get_future_predictions(junction_id):
    predictions = generate_future_predictions(junction_id)
    if predictions:
        return jsonify(predictions)
    return jsonify([])
# Add this new endpoint to get individual junction data
@app.route('/get_junction_data/<int:junction_id>')
def get_junction_data(junction_id):
    latest_data = TrafficData.query.filter_by(
        junction_id=junction_id,
        analysis_completed=True
    ).order_by(TrafficData.created_at.desc()).first()
    
    if not latest_data:
        return jsonify({"error": "No data available for this junction"}), 404
    
    return jsonify({
        "original_image": latest_data.original_image_path,
        "processed_image": latest_data.processed_image_path,
        "total_vehicles": latest_data.total_vehicles,
        "congestion_level": latest_data.congestion_level,
        "congestion_percentage": latest_data.congestion_percentage,
        "signal_timings": latest_data.signal_timings,
        "vehicle_counts": latest_data.vehicle_counts,
        "timestamp": latest_data.created_at.isoformat()
    })

# Modify the analyze_junction endpoint to return more detailed data
# (Keep the existing code but enhance the return JSON)
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        ensure_columns_exist()
    app.run(debug=True, threaded=True)