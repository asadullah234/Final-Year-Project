from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from sqlalchemy import inspect, text

app = Flask(__name__, static_url_path='/static')

# Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:asad@localhost:5432/Traffic-Congestion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'mp4'}

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/realtime')
def realtime():
    return render_template('realtime_result.html')

@app.route('/analyze_junction', methods=['POST'])
def analyze_junction():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    junction_id = int(request.form.get('junction_id', 1))
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    try:
        filename = secure_filename(f"junction_{junction_id}_{int(time.time())}.jpg")
        original_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(original_path)
        
        result = analyze_image(original_path, junction_id)
        if "error" in result:
            os.remove(original_path)
            return jsonify({"error": result["error"]}), 400
        
        # Save to database
        data = TrafficData(
            junction_id=junction_id,
            original_image_path=filename,
            processed_image_path=result["processed_image"],
            vehicle_counts=result["counts"],
            total_vehicles=result["total"],
            congestion_level=result["congestion_level"],
            congestion_percentage=result["congestion_pct"],
            signal_timings=result["signal_timings"],
            analysis_completed=True
        )
        db.session.add(data)
        db.session.commit()
        
        return jsonify({
            "success": True,
            "junction_id": junction_id,
            "result": result
        })
    
    except Exception as e:
        db.session.rollback()
        if 'original_path' in locals() and os.path.exists(original_path):
            os.remove(original_path)
        if 'result' in locals() and 'processed_image' in result:
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], result["processed_image"])
            if os.path.exists(processed_path):
                os.remove(processed_path)
        return jsonify({"error": str(e)}), 500

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

@app.route('/get_junction_status')
def get_junction_status():
    status = {}
    for junction_id in range(1, 5):
        latest = TrafficData.query.filter_by(junction_id=junction_id).order_by(TrafficData.created_at.desc()).first()
        status[junction_id] = {
            "completed": latest.analysis_completed if latest else False,
            "timestamp": latest.created_at.isoformat() if latest else None
        }
    return jsonify(status)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        ensure_columns_exist()
    app.run(debug=True, threaded=True)