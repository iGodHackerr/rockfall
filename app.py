import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory
from sklearn.ensemble import RandomForestRegressor
from perlin_noise import PerlinNoise

# --- App Initialization ---
app = Flask(__name__, template_folder='.', static_folder='.')

# --- Constants ---
GRID_SIZE = 20
MODEL_FILE = 'rockfall_model.pkl'
STATIC_DATA_FILE = 'static_mine_data.csv'
PREDICTION_FEATURES = [
    'slope_angle', 'rock_quality', 'displacement', 'strain', 'pore_pressure',
    'rainfall', 'temperature', 'vibration'
]


# --- Synthetic Data and Model Generation ---
def generate_synthetic_data():
    """Generates and saves synthetic mine data and a prediction model if they don't exist."""
    
    print("Forcing regeneration of data and model for improved realism...")
    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
    if os.path.exists(STATIC_DATA_FILE): os.remove(STATIC_DATA_FILE)

    print("Generating synthetic data and training model...")
    
    # 1. Generate Static Mine Data
    cell_ids = [f'cell-{i}-{j}' for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    num_cells = GRID_SIZE * GRID_SIZE
    
    slope_angles = np.random.uniform(20, 45, num_cells)
    rock_quality = np.random.randint(5, 9, num_cells)
    base_displacement = np.random.uniform(0.1, 0.5, num_cells)
    base_strain = np.random.uniform(50, 250, num_cells)
    base_pore_pressure = np.random.uniform(2, 10, num_cells)

    center_i, center_j = GRID_SIZE // 2, GRID_SIZE // 2
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            distance = np.sqrt((i - center_i)**2 + (j - center_j)**2)
            if distance < GRID_SIZE / 3:
                slope_angles[idx] = np.random.uniform(55, 75)
                rock_quality[idx] = np.random.randint(1, 4)
                base_displacement[idx] = np.random.uniform(0.8, 1.5)
                base_strain[idx] = np.random.uniform(400, 800)
                base_pore_pressure[idx] = np.random.uniform(15, 30)
            
    static_data = pd.DataFrame({
        'cell_id': cell_ids, 'slope_angle': np.round(slope_angles, 2), 'rock_quality': rock_quality,
        'base_displacement': np.round(base_displacement, 2), 'base_strain': np.round(base_strain),
        'base_pore_pressure': np.round(base_pore_pressure, 2)
    })
    static_data.to_csv(STATIC_DATA_FILE, index=False)

    # 2. Generate Training Data with non-linear risk escalation
    num_samples = 25000
    training_data = {
        'slope_angle': np.random.uniform(20, 80, num_samples), 'rock_quality': np.random.randint(1, 10, num_samples),
        'displacement': np.random.uniform(0.1, 5.0, num_samples), 'strain': np.random.uniform(50, 2000, num_samples),
        'pore_pressure': np.random.uniform(2, 60, num_samples), 'rainfall': np.random.uniform(0, 100, num_samples),
        'temperature': np.random.uniform(-10, 40, num_samples), 'vibration': np.random.randint(0, 2, num_samples),
    }
    
    base_risk = (training_data['slope_angle'] / 90.0) * 0.05 + (1 - (training_data['rock_quality'] / 10.0)) * 0.05
    dynamic_risk = np.power(training_data['displacement'] / 5.0, 2.5) * 0.30 + \
                   np.power(training_data['strain'] / 2000.0, 2.5) * 0.20 + \
                   np.power(training_data['pore_pressure'] / 60.0, 2) * 0.25 + \
                   np.power(training_data['rainfall'] / 100.0, 2) * 0.15 + \
                   (training_data['vibration']) * 0.20
    
    total_risk = (base_risk + dynamic_risk) * 1.5
    risk = np.clip(total_risk + np.random.normal(0, 0.02, num_samples), 0, 1)
    
    training_df = pd.DataFrame(training_data)
    training_df['risk'] = risk

    # 3. Train and Save Model
    X = training_df[PREDICTION_FEATURES]
    y = training_df['risk']
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=5)
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)
    print("Model training complete.")

# --- Load Model and Data ---
generate_synthetic_data()
model = joblib.load(MODEL_FILE)
static_mine_data_df = pd.read_csv(STATIC_DATA_FILE)
static_mine_data_indexed = static_mine_data_df.set_index('cell_id')
print("Model and static data loaded.")

# --- Perlin Noise Generators for Realistic Variation ---
noise_rainfall = PerlinNoise(octaves=5, seed=np.random.randint(1, 500))
noise_displacement = PerlinNoise(octaves=6, seed=np.random.randint(501, 1000))
noise_pressure = PerlinNoise(octaves=4, seed=np.random.randint(1001, 1500))
noise_strain = PerlinNoise(octaves=7, seed=np.random.randint(1501, 2000))


# --- API Routes ---
@app.route('/')
def index():
    return render_template('index.html')

# --- NEW ROUTE TO SERVE THE IMAGE ---
@app.route('/<filename>')
def get_image(filename):
    """Serves the background image from the root directory."""
    return send_from_directory('.', filename)

@app.route('/api/mine-static-data')
def get_mine_static_data():
    return jsonify(static_mine_data_df.to_dict(orient='records'))

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction_df = static_mine_data_indexed.copy()

        noise_map_rainfall = np.array([noise_rainfall([i/GRID_SIZE, j/GRID_SIZE]) for i in range(GRID_SIZE) for j in range(GRID_SIZE)])
        noise_map_displacement = np.array([noise_displacement([i/GRID_SIZE, j/GRID_SIZE]) for i in range(GRID_SIZE) for j in range(GRID_SIZE)])
        noise_map_pressure = np.array([noise_pressure([i/GRID_SIZE, j/GRID_SIZE]) for i in range(GRID_SIZE) for j in range(GRID_SIZE)])
        noise_map_strain = np.array([noise_strain([i/GRID_SIZE, j/GRID_SIZE]) for i in range(GRID_SIZE) for j in range(GRID_SIZE)])

        def rescale_noise_to_01(noise_array):
            return (noise_array + 0.7) / 1.4

        prediction_df['rainfall'] = float(data['rainfall']) * rescale_noise_to_01(noise_map_rainfall)
        prediction_df['displacement'] = float(data['displacement']) * rescale_noise_to_01(noise_map_displacement)
        prediction_df['pore_pressure'] = float(data['pore_pressure']) * rescale_noise_to_01(noise_map_pressure)
        prediction_df['strain'] = float(data['strain']) * rescale_noise_to_01(noise_map_strain)
        
        prediction_df['temperature'] = float(data['temperature'])
        prediction_df['vibration'] = int(data['vibration'])
        
        for col in ['rainfall', 'displacement', 'pore_pressure', 'strain']:
            prediction_df[col] = prediction_df[col].clip(lower=0)
        prediction_df = prediction_df[PREDICTION_FEATURES]

        predictions = model.predict(prediction_df)
        results = {cell_id: risk for cell_id, risk in zip(prediction_df.index, predictions)}
        
        return jsonify(results)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)

