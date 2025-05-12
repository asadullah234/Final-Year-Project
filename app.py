#importing lib
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import torch
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
import base64
from collections import Counter
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require a GUI
import matplotlib.pyplot as plt
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/processed'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:asad@localhost:5432/TrafficCongestion'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

class TrafficData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    original_image_path = db.Column(db.String, nullable=False)
    processed_image_path = db.Column(db.String, nullable=False)
    vehicle_counts = db.Column(db.JSON, nullable=False)  # Store counts as JSON
    total_vehicles = db.Column(db.Integer, nullable=False)
    signal_timings = db.Column(db.JSON, nullable=False)  # Store timings as JSON
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<TrafficData {self.id}>'

class TrafficSignalAnalyzer:
    def __init__(self, model_path="best.pt"):
        """Initialize the traffic signal analyzer.
        
        Args:
            model_path: Path to the YOLO model weights
        """
        self.model = None # Initialize model attribute
        # Load YOLO model
        try:
            self.model = YOLO(model_path)
            print(f"Successfully loaded YOLO model from {model_path}")
            if self.model and hasattr(self.model, 'names'):
                 print(f"DEBUG: Model class names available from self.model.names: {self.model.names}")
            else:
                 print(f"DEBUG: Model loaded, but 'names' attribute not found or model is None for path {model_path}.")
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            print("Continuing without YOLO model functionality...")
            self.model = None # Ensure model is None if loading failed
        
        # Define vehicle classes for display and as keys in our counts dict
        # These are the canonical names we will use.
        self.display_vehicle_classes = ['Bike', 'Bus', 'Rickshaw', 'Truck', 'car', 'pickup']
        
        # Create a mapping from potential model output (lowercase) to our canonical display names
        # This handles case variations from the model's output.
        self.model_output_to_display_map = {name.lower(): name for name in self.display_vehicle_classes}
        
        # Initialize counts using the canonical display names as keys
        self.vehicle_counts_template = {name: 0 for name in self.display_vehicle_classes} # For consistent initialization
        self.vehicle_counts = self.vehicle_counts_template.copy() # Instance variable for current counts
        self.total_vehicles = 0
        
    def detect_vehicles_from_image(self, image_path):
        """Detect vehicles in an image using YOLO.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with vehicle counts and processed image path
        """
        # Read image from path
        img_read = cv2.imread(image_path) # Use a different variable name to avoid confusion
        if img_read is None:
            return {"error": f"Failed to load image from {image_path}"}
        
        # Make a copy to draw on, preserving the original if needed elsewhere (though not strictly needed here)
        image_to_process = img_read.copy()
        
        # Initialize vehicle counts dictionary with zeros using canonical display names as keys
        current_image_vehicle_counts = self.vehicle_counts_template.copy()
        
        if self.model is None:
            print("Warning: YOLO model is not loaded. Returning zero counts and original image.")
            # Save the original image as processed if no model, to maintain consistent output structure
            output_filename = 'processed_no_model_' + os.path.basename(image_path)
            output_path = os.path.join(os.path.dirname(image_path), output_filename)
            cv2.imwrite(output_path, image_to_process) # Save the unprocessed image copy
            return {
                "vehicle_count": current_image_vehicle_counts, # All zeros
                "total_vehicles": 0,
                "processed_image_path": output_path # Return path to the (un)processed image
            }
        
        # Run inference
        results = self.model(image_to_process) # Process the copy
        
        # Process detection results
        for result_item in results: # Renamed 'result' to 'result_item' for clarity
            for box in result_item.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                class_id = int(box.cls[0])               # Class ID
                confidence = box.conf[0].item()          # Confidence score
                
                # Get class name using the model's class names
                raw_class_name_from_model = self.model.names[class_id]
                normalized_model_output = raw_class_name_from_model.lower() # Normalize to lowercase for lookup

                # Find the canonical display name that corresponds to the normalized model output
                canonical_name = self.model_output_to_display_map.get(normalized_model_output)
                                
                if canonical_name:
                    current_image_vehicle_counts[canonical_name] += 1
                else:
                    # This handles cases where the model might output something unexpected
                    print(f"Warning: Detected class '{raw_class_name_from_model}' (normalized: '{normalized_model_output}') from model is not mapped to any display class. Not counted.")
                
                # Draw bounding box
                cv2.rectangle(image_to_process, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label (use the raw_class_name_from_model for display for original casing)
                label = f"{raw_class_name_from_model} {confidence:.2f}"
                cv2.putText(image_to_process, label, (x1, y1 - 10), # Adjusted y-offset for label
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calculate total vehicles for the current image
        current_image_total_vehicles = sum(current_image_vehicle_counts.values())
        
        # Update instance counts (reflects the last processed image) - Optional if not used elsewhere
        self.vehicle_counts = current_image_vehicle_counts
        self.total_vehicles = current_image_total_vehicles
        
        # Save the processed image
        output_filename = 'processed_' + os.path.basename(image_path)
        output_path = os.path.join(os.path.dirname(image_path), output_filename)
        cv2.imwrite(output_path, image_to_process) # Save the image with drawings
        
        return {
            "vehicle_count": current_image_vehicle_counts, # Counts for this specific image
            "total_vehicles": current_image_total_vehicles, # Total for this specific image
            "processed_image_path": output_path
        }
    
    def get_counts(self):
        """Get current vehicle counts (reflects last processed image or initial state)."""
        return {
            "vehicle_count": self.vehicle_counts,
            "total_vehicles": self.total_vehicles
        }

class TrafficSignalController:
    def __init__(self):
        """Initialize the traffic signal controller with default values."""
        # Default signal times
        self.green_time = 5  # Default green time
        self.yellow_time = 3  # Yellow time is fixed
        self.red_time = 20    # Default red time
        
        # Threshold for vehicle congestion (100% congestion)
        self.max_vehicle_threshold = 30

    def calculate_congestion_percentage(self, total_vehicles):
        """Calculate congestion percentage based on vehicle count.
        
        Args:
            total_vehicles: Total number of vehicles detected
            
        Returns:
            Congestion percentage (0-100)
        """
        if self.max_vehicle_threshold == 0: # Avoid division by zero
            return 100 if total_vehicles > 0 else 0
        congestion = (total_vehicles / self.max_vehicle_threshold) * 100
        # Cap at 100%
        return min(congestion, 100)

    def update_signal_timings(self, vehicle_data):
        """
        Dynamically update signal timings based on vehicle count
        
        Args:
            vehicle_data: Dict with vehicle count information
        """
        # Extract total vehicle count
        total_vehicles = vehicle_data.get("total_vehicles", 0)
        
        # Calculate congestion percentage
        congestion_percentage = self.calculate_congestion_percentage(total_vehicles)
        
        # Set green time according to requirements
        if total_vehicles < 5:
            self.green_time = 5
        elif total_vehicles < 10:
            self.green_time = 10
        elif total_vehicles < 20:
            self.green_time = 15
        elif total_vehicles < 30:
            self.green_time = 30 # As per original logic, this was 30, then 45 for 30+
        else: # total_vehicles >= 30
            self.green_time = 45 + ((total_vehicles - 30) // 10) * 15
            self.green_time = min(self.green_time, 120)
        
        self.red_time = max(20, self.green_time - 5) # Ensure red time is at least 20
        
        timings = self.get_signal_timings()
        timings.update({
            "congestion_percentage": round(congestion_percentage, 1)
        })
        
        return timings

    def get_signal_timings(self):
        """Get current signal timings."""
        return {
            "green_time": self.green_time,
            "yellow_time": self.yellow_time,
            "red_time": self.red_time
        }

def create_vehicle_count_chart(vehicle_counts):
    """Create a bar chart of vehicle counts and return as base64 image"""
    if not vehicle_counts or all(count == 0 for count in vehicle_counts.values()):
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No vehicles detected", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_str
    
    filtered_counts = {k: v for k, v in vehicle_counts.items() if v > 0}
    
    if not filtered_counts:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No vehicles detected (after filtering zeros)", horizontalalignment='center', 
                 verticalalignment='center', transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_str

    plt.figure(figsize=(10, 6))
    sorted_counts = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [item[0] for item in sorted_counts]
    values = [item[1] for item in sorted_counts]
    colors = plt.cm.get_cmap('tab10', len(labels))
    plt.bar(labels, values, color=colors.colors)
    plt.xlabel('Vehicle Type')
    plt.ylabel('Count')
    plt.title('Vehicle Counts by Type')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return img_str

def create_line_chart(results):
    """Create a line chart showing vehicle count per junction"""
    has_vehicles = any(result.get('total_vehicles', 0) > 0 for result in results)
    
    if not has_vehicles or not results:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No vehicles detected for comparison", 
                 horizontalalignment='center', verticalalignment='center', 
                 transform=plt.gca().transAxes, fontsize=14)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()
        return img_str
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    junctions = [f'Junction {i+1}' for i in range(len(results))]
    vehicle_counts_data = [result['total_vehicles'] for result in results]
    congestion_percentages = [result['timings']['congestion_percentage'] for result in results]
    green_times = [result['timings']['green_time'] for result in results]
        
    ax1.plot(junctions, vehicle_counts_data, 'o-', color='blue', linewidth=2, markersize=8)
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Traffic Analysis by Junction')
    ax1.grid(True, linestyle='--', alpha=0.7)
    for i, count in enumerate(vehicle_counts_data):
        ax1.annotate(str(count), (junctions[i], count), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    ax2.plot(junctions, congestion_percentages, 'o-', color='red', linewidth=2, markersize=8, label='Congestion %')
    ax2.set_ylabel('Congestion %', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.grid(True, linestyle='--', alpha=0.7)
    for i, pct in enumerate(congestion_percentages):
        ax2.annotate(f"{pct}%", (junctions[i], pct), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', color='red')
    
    ax3 = ax2.twinx()
    ax3.plot(junctions, green_times, 's-', color='green', linewidth=2, markersize=8, label='Green Time (s)')
    ax3.set_ylabel('Green Time (seconds)', color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    for i, time_val in enumerate(green_times):
        ax3.annotate(f"{time_val}s", (junctions[i], time_val), 
                    textcoords="offset points", 
                    xytext=(0,-15),
                    ha='center', color='green')
    
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax2.set_xlabel('Junction')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_str

def analyze_traffic_efficiency(results):
    """Analyze overall traffic efficiency across junctions"""
    if not results:
        return {
            'total_vehicles': 0,
            'avg_green_time': 0,
            'avg_vehicles_per_junction': 0,
            'avg_congestion': 0,
            'total_vehicle_types': {},
            'no_data': True
        }
    
    aggregated_vehicle_types = Counter()
    for result in results:
        counts_for_junction = result.get('vehicle_counts', {})
        if isinstance(counts_for_junction, dict):
            aggregated_vehicle_types.update(counts_for_junction)
        else:
            print(f"Warning: vehicle_counts for a result was not a dict: {counts_for_junction}")

    total_vehicles_overall = sum(result.get('total_vehicles', 0) for result in results)
    
    if total_vehicles_overall == 0:
        return {
            'total_vehicles': 0,
            'avg_green_time': sum(result['timings']['green_time'] for result in results) / len(results) if results else 0,
            'avg_vehicles_per_junction': 0,
            'avg_congestion': sum(result['timings']['congestion_percentage'] for result in results) / len(results) if results else 0,
            'total_vehicle_types': dict(aggregated_vehicle_types),
            'no_vehicles': True
        }
    
    avg_green_time = sum(result['timings']['green_time'] for result in results) / len(results)
    avg_total_vehicles_per_junction = total_vehicles_overall / len(results)
    avg_congestion = sum(result['timings']['congestion_percentage'] for result in results) / len(results)
    
    efficiency_metrics = {
        'total_vehicles': total_vehicles_overall,
        'avg_green_time': round(avg_green_time, 2),
        'avg_vehicles_per_junction': round(avg_total_vehicles_per_junction, 2),
        'avg_congestion': round(avg_congestion, 2),
        'total_vehicle_types': dict(aggregated_vehicle_types)
    }
        
    return efficiency_metrics

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

analyzer = TrafficSignalAnalyzer()
controller = TrafficSignalController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(file.filename == '' for file in uploaded_files):
        return redirect(url_for('index'))
    
    all_results_data = []
    valid_files_processed = 0

    for file in uploaded_files[:4]:
        if file and file.filename != '' and allowed_file(file.filename):
            filename = str(int(time.time())) + '_' + Path(file.filename).name
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            detection_result = analyzer.detect_vehicles_from_image(filepath)
            if "error" in detection_result:
                print(f"Error processing {filename}: {detection_result['error']}")
                continue
            
            valid_files_processed += 1
            timings = controller.update_signal_timings(detection_result)
            
            traffic_data = TrafficData(
                original_image_path=filepath,
                processed_image_path=detection_result.get("processed_image_path", ""),
                vehicle_counts=detection_result.get("vehicle_count", {}),
                total_vehicles=detection_result.get("total_vehicles", 0),
                signal_timings=timings
            )
            db.session.add(traffic_data)
            db.session.commit()
            
            all_results_data.append({
                "original_image": filepath.replace('static/', '', 1),
                "processed_image": detection_result.get("processed_image_path", "").replace('static/', '', 1),
                "vehicle_counts": detection_result.get("vehicle_count", {}),
                "total_vehicles": detection_result.get("total_vehicles", 0),
                "timings": timings,
                "junction_name": f"Junction {len(all_results_data) + 1}"
            })
    
    if not valid_files_processed:
        return redirect(url_for('index'))
    
    has_vehicles_overall = any(res.get('total_vehicles', 0) > 0 for res in all_results_data)
    line_chart_img = create_line_chart(all_results_data) if len(all_results_data) > 0 else None
    traffic_efficiency_metrics = analyze_traffic_efficiency(all_results_data)
    
    return render_template('multi_result.html', 
                           results=all_results_data, 
                           line_chart=line_chart_img,
                           traffic_efficiency=traffic_efficiency_metrics,
                           has_vehicles=has_vehicles_overall)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    uploaded_files = request.files.getlist('files')
    if not uploaded_files or all(file.filename == '' for file in uploaded_files):
        return jsonify({"error": "No files or empty file list provided"}), 400
    
    api_results_data = []
    valid_files_processed_api = 0

    for file in uploaded_files[:4]:
        if file and file.filename != '' and allowed_file(file.filename):
            filename = str(int(time.time())) + '_' + Path(file.filename).name
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            detection_result = analyzer.detect_vehicles_from_image(filepath)
            if "error" in detection_result:
                print(f"API Error processing {filename}: {detection_result['error']}")
                continue
            
            valid_files_processed_api += 1
            timings = controller.update_signal_timings(detection_result)
            
            traffic_data = TrafficData(
                original_image_path=filepath,
                processed_image_path=detection_result.get("processed_image_path", ""),
                vehicle_counts=detection_result.get("vehicle_count", {}),
                total_vehicles=detection_result.get("total_vehicles", 0),
                signal_timings=timings
            )
            db.session.add(traffic_data)
            db.session.commit()
            
            api_results_data.append({
                "original_image_path": filepath,
                "processed_image_path": detection_result.get("processed_image_path", ""),
                "vehicle_counts": detection_result.get("vehicle_count", {}),
                "total_vehicles": detection_result.get("total_vehicles", 0),
                "signal_timings": timings,
            })
    
    if not valid_files_processed_api:
        return jsonify({"error": "No valid image files processed"}), 400
        
    return jsonify({"results": api_results_data})

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("Created 'templates' directory. Make sure your HTML files (index.html, multi_result.html) are in it.")

    app.run(debug=True)

