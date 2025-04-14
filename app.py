# Import libraries
import os
import re
import pickle
import csv
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, flash, url_for
from flask_sqlalchemy import SQLAlchemy

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Needed for flash messages

# Database configuration for SQLite
basedir = os.path.abspath(os.path.dirname(__file__))
os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)  # Ensure instance folder exists
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'heart_data.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Load ML models
try:
    all_models = pickle.load(open('models.pkl', 'rb'))
    all_models2 = pickle.load(open('models.pkl', 'rb'))
except FileNotFoundError:
    raise Exception("Model files not found. Please ensure models.pkl exists in your project directory.")

# Define Patient Data Model
class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100))
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    cp = db.Column(db.Integer)  # chest pain type
    trestbps = db.Column(db.Integer)  # resting blood pressure
    chol = db.Column(db.Integer)  # cholesterol
    fbs = db.Column(db.Integer)  # fasting blood sugar
    restecg = db.Column(db.Integer)  # resting electrocardiographic results
    thalach = db.Column(db.Integer)  # maximum heart rate achieved
    exang = db.Column(db.Integer)  # exercise induced angina
    oldpeak = db.Column(db.Float)  # ST depression induced by exercise
    slope = db.Column(db.Integer)  # slope of peak exercise ST segment
    ca = db.Column(db.Integer)  # number of major vessels
    thal = db.Column(db.Integer)  # thalassemia
    prediction_results = db.Column(db.String(500))  # store all model predictions
    average_risk = db.Column(db.Float)  # store the average accuracy/risk
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<PatientRecord {self.name}>'

# Create tables (run this once)
with app.app_context():
    db.create_all()

# CSV file configuration
CSV_FILE_PATH = os.path.join('dataset', 'heart_data_set.csv')
CSV_HEADERS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
              'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Ensure CSV file exists with headers
if not os.path.exists(CSV_FILE_PATH):
    with open(CSV_FILE_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        writer.writerow(CSV_HEADERS)

@app.route('/', methods=['GET', 'POST'])
def hello():
    return render_template("index.html")

@app.route('/aboutUs', methods=['GET'])
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/api', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data with validation
            name = request.form['name'].strip()
            if not name:
                flash('Name is required', 'error')
                return redirect(url_for('hello'))
                
            email = request.form['email'].strip()
            if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                flash('Invalid email address', 'error')
                return redirect(url_for('hello'))
                
            age = int(request.form['age'])
            if not (5 <= age <= 100):
                flash('Age must be between 5 and 100', 'error')
                return redirect(url_for('hello'))
                
            # Get other form data
            fgender = request.form['gender']
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps']) if request.form['trestbps'] else 95
            chol = int(request.form['chol']) if request.form['chol'] else 150
            thalach = int(request.form['thalach']) if request.form['thalach'] else 72
            oldpeak = float(request.form['oldpeak']) if request.form['oldpeak'] else 2.0
            fbs = int(request.form['fbs'])
            restecg = request.form['restecg']
            exang = request.form['exang']
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = request.form['thal']

            # Convert form data for model prediction
            gender = 1 if fgender == "Male" else 0
            
            if thal == "Normal":
                thal = 0
            elif thal == "Fixed Defect":
                thal = 1
            else:
                thal = 2

            if restecg == "Normal":
                restecg = 0
            elif restecg == "STT Abnormality":
                restecg = 1
            else:
                restecg = 2

            exang = 1 if exang == "Yes" else 0

            # Prepare features for model prediction
            features = [
                age, gender, cp, trestbps, chol, 
                fbs, restecg, thalach, exang, 
                oldpeak, slope, ca, thal
            ]

            # Get predictions from all models
            predictions = {}
            avg = 0
            
            for model in all_models:
                res = model.predict([features])[0]
                predictions[str(model)] = "High Chance of Heart Disease" if res == 1 else "Low Chance of Heart Disease"
                avg += res

            # Calculate average risk
            accuracy = round(avg/len(all_models), 2)
            target = 1 if accuracy > 0.5 else 0

            # Store data in database
            new_record = PatientRecord(
                name=name,
                email=email,
                age=age,
                gender=fgender,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
                prediction_results=str(predictions),
                average_risk=accuracy
            )

            db.session.add(new_record)
            db.session.commit()

            # Prepare CSV data in exact format as original
            csv_data = [
                age, 
                gender,  # sex (1=male, 0=female)
                cp,
                trestbps,
                chol,
                fbs,
                restecg,
                thalach,
                exang,
                oldpeak,
                slope,
                ca,
                thal,
                target  # 0 or 1 based on risk
            ]
            
            # Write to CSV with space delimiter to match original format
            try:
                with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=' ')
                    writer.writerow(csv_data)
            except Exception as e:
                print(f"Error writing to CSV: {str(e)}")
                # You might want to log this error properly in production

            # Prepare response data
            input_data = {
                "age": age,
                "Gender": fgender,
                "Chest Pain Types": cp,
                "Resting Blood Pressure(in mm/Hg)": trestbps,
                "Cholesterol Level": chol,
                "is Fasting Blood Pressure>120mg/Dl?": fbs,
                "Resting Electro Cardio Graphic Result": restecg,
                "Maximum Heart Rate Achieved": thalach,
                "Does Exercise Induced Angina?": exang,
                "Old Peak (ST Depression Induced by Exercise Relative to Rest)": oldpeak,
                "Slope of ST Segment": slope,
                "number of major vessels (0-3) colored by flourosopy": ca,
                "Thal Type": thal
            }

            personal_info = [name, email]
            responses = [input_data, predictions, personal_info, accuracy]

            return render_template("result.html", result=responses)

        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred: {str(e)}', 'error')
            return redirect(url_for('hello'))

    return redirect(url_for('hello'))

@app.route('/view-data')
def view_data():
    with app.app_context():
        records = PatientRecord.query.order_by(PatientRecord.created_at.desc()).all()
        return render_template('view_data.html', records=records)

if __name__ == '__main__':
    app.run(port=5000, debug=True)