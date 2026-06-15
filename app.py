import os 
import pickle 
import json 
import pandas as pd 
import numpy as np 
import shap 
from flask import Flask, request, render_template, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret_key'


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login' 


basedir = os.path.abspath(os.path.dirname(__file__))

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


os.makedirs(os.path.join(basedir, 'instance'), exist_ok=True)


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()
    



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_models():
    try:
        
        model_path = os.path.join(BASE_DIR, "models", "xgb_model.pkl")
        male_km_path = os.path.join(BASE_DIR, "models", "km_male.pkl")
        female_km_path = os.path.join(BASE_DIR, "models", "km_female.pkl")

        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(male_km_path, "rb") as male_clusterer_file:
            km_male = pickle.load(male_clusterer_file)
        with open(female_km_path, "rb") as female_clusterer_file:
            km_female = pickle.load(female_clusterer_file)
            
        return model, km_male, km_female
    except FileNotFoundError:
        return None, None, None

model, km_male, km_female = load_models()


def preprocess_input(input_data_dict, selected_gender_for_kmodes, km_male_model, km_female_model):
    df = pd.DataFrame([input_data_dict])
    df['BMI'] = df['weight'] / (df['height'] / 100) ** 2
    df['map'] = ((2 * df['ap_lo']) + df['ap_hi']) / 3

    age_bins = [30, 35, 40, 45, 50, 55, 60, 65, np.inf]
    age_labels = list(range(len(age_bins) - 1))
    df['age'] = pd.cut(df['age(years)'], bins=age_bins, labels=age_labels)

    bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_labels = [0, 1, 2, 3, 4, 5]
    df['BMI'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)

    map_bins = [0, 70, 80, 90, 100, 110, np.inf]
    map_labels = [0, 1, 2, 3, 4, 5]
    df['map'] = pd.cut(df['map'], bins=map_bins, labels=map_labels)

    df = df.drop(['age(years)', 'height', 'weight', 'ap_hi', 'ap_lo'], axis=1)
    df[['age', 'BMI', 'map']] = df[['age', 'BMI', 'map']].fillna(0)
    for col in ['age', 'BMI', 'map']:
        df[col] = df[col].astype(int)

    cluster_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'BMI', 'map', 'age']
    user_df_for_cluster = df[cluster_features].copy()
    user_df_for_cluster['cluster'] = 2

    if df['gender'].iloc[0] == 2:
        df['cluster'] = km_male_model.predict(user_df_for_cluster)
    elif df['gender'].iloc[0] == 1:
        df['cluster'] = km_female_model.predict(user_df_for_cluster)
    else:
        df['cluster'] = 0

    df = df[['gender', 'cholesterol', 'smoke', 'alco', 'gluc', 'active', 'BMI', 'map', 'age', 'cluster']]
    return df




@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:

            return redirect(url_for('register'))

       
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = User.query.filter_by(username=username).first()
        
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
            

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/', methods=['GET', 'POST'])
@login_required 
def index():
    if model is None or km_male is None or km_female is None:
        return render_template('index.html', error="Model or KModes clusterer files not found.")

    if request.method == 'POST':
        try:
           
            user_data = {
                "age(years)": float(request.form['age_years']),
                "gender": int(request.form['gender']),
                "height": float(request.form['height']),
                "weight": float(request.form['weight']),
                "ap_hi": float(request.form['ap_hi']),
                "ap_lo": float(request.form['ap_lo']),
                "cholesterol": int(request.form['cholesterol']),
                "gluc": int(request.form['gluc']),
                "smoke": int(request.form['smoke']),
                "alco": int(request.form['alco']),
                "active": int(request.form['active']),
            }
            
            processed_data = preprocess_input(user_data, user_data['gender'], km_male, km_female)
            prediction = model.predict(processed_data)[0]
            prediction_proba = model.predict_proba(processed_data)[:, 1][0]

            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_data)
            feature_names = processed_data.columns
            feature_names_map = {
                'gender': 'Gender',
                'cholesterol': 'Cholesterol',
                'smoke': 'Smoking',
                'alco': 'Alcohol',
                'gluc': 'Glucose',
                'active': 'Physical Activity',
                'BMI': 'Weight (BMI)',
                'map': 'Blood Pressure (MAP)',
                'age': 'Age',
                'cluster': 'Cluster'
            }
            shap_dict = {feature: float(shap_values[0][i]) for i, feature in enumerate(feature_names)}
            shap_dict.pop('cluster')
            shap_dict.pop('gender')
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            importance_data = [(feature_names_map.get(k, k), f"{v:.3f}") for k, v in  sorted_shap]

            
            chart_data = {
                "labels": [feature_names_map.get(k, k) for k, v in  sorted_shap],
                "values": [float(abs(v)) for k, v in  sorted_shap]  
            }

            
 
            feature_recommendations = {
                "cholesterol": "Monitor and manage cholesterol levels.",
                "gluc": "Monitor and manage blood glucose levels.",
                "smoke": "Quit smoking immediately.",
                "alco": "Reduce or eliminate alcohol consumption.",
                "active": "Increase your physical activity.",
                "BMI": "Follow a heart-healthy diet low in saturated fats and cholesterol.",
                "map": "Monitor and manage your blood pressure regularly.",
                "age": "Get regular health checkups due to age-related risk.",
            }

            recs = []

          
            if prediction == 1:
                recs.append("Consult with a cardiologist for comprehensive evaluation.")

         
            for feature, recommendation in feature_recommendations.items():
                shap_value = shap_dict.get(feature, 0)
                if shap_value > 0:
                    recs.append(recommendation)

            return render_template(
                'index.html',
                result="High Risk" if prediction == 1 else "Low Risk",
                risk_probability=f"{prediction_proba*100:.1f}%",
                importance_data=importance_data,
                recommendations=recs,
                chart_data=json.dumps(chart_data)
            )
        except Exception as e:
            return render_template('index.html', error=f"Prediction error: {e}")

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)