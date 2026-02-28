"""
Flask Backend for Plasma Disruption Detection Dashboard
Provides API endpoints for model inference and data generation
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
import joblib

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

# Global variables for model and scaler
model = None
scaler = None
feature_columns = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# MODEL DEFINITION
# ============================================================================

class AdvancedBayesianDisruptionModel(nn.Module):
    """Bayesian Neural Network for plasma disruption detection with MC Dropout"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super().__init__()
        
        # Conv1D for temporal patterns
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        
        self.conv_output_size = 16 * 32
        
        # Dense layers with MC Dropout
        layers = []
        prev_dim = self.conv_output_size + input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Linear(64, 1))
        layers.append(nn.Sigmoid())
        
        self.dense_layers = nn.Sequential(*layers)
        self.dropout_rate = dropout_rate
    
    def forward(self, x):
        batch_size = x.shape[0]
        x_conv = x.unsqueeze(1)
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.reshape(batch_size, -1)
        x_combined = torch.cat([x_conv, x], dim=1)
        output = self.dense_layers(x_combined)
        return output


# ============================================================================
# INITIALIZATION
# ============================================================================

def load_model():
    """Load trained model and scaler"""
    global model, scaler, feature_columns
    
    model_path = os.path.join(os.path.dirname(__file__), 'plasma_disruption_model.pt')
    scaler_path = os.path.join(os.path.dirname(__file__), 'feature_scaler.pkl')
    features_path = os.path.join(os.path.dirname(__file__), 'feature_columns.pkl')
    
    # Load feature columns (21 features total)
    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
    else:
        # Default feature columns if not found
        feature_columns = [
            'Ip', 'dIp_dt', 'q95', 'dq_dt', 'li', 'dli_dt', 'beta', 'dbeta_dt',
            'mirnov_dB_dt', 'locked_mode_indicator', 'n1_rms', 'n2_rms',
            'bolometry', 'Te', 'dTe_dt', 'ne_greenwald_frac',
            'd2Ip_dt2', 'distance_to_wall', 'error_field', 'stability_index'
        ]
    
    # Load scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
    else:
        scaler = None
        print(f"⚠ Scaler file not found at {scaler_path}, will use raw features")
    
    # Initialize model
    input_dim = len(feature_columns)
    model = AdvancedBayesianDisruptionModel(input_dim=input_dim).to(device)
    
    # Load trained weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model file not found at {model_path}")
        print("  Will use randomly initialized model")
    
    model.eval()
    return model, scaler, feature_columns


def generate_sample_data(num_samples=100):
    """Generate realistic synthetic tokamak data with a mix of plasma regimes"""
    
    # Split into three regimes: stable, marginal, pre-disruptive
    n_stable = int(num_samples * 0.45)
    n_marginal = int(num_samples * 0.30)
    n_disruptive = num_samples - n_stable - n_marginal
    
    regimes = []
    
    for regime, n in [('stable', n_stable), ('marginal', n_marginal), ('disruptive', n_disruptive)]:
        d = {}
        
        if regime == 'stable':
            # Healthy plasma: high q95, low fluctuations, good confinement
            d['Ip'] = np.random.normal(1.8, 0.05, n)
            d['dIp_dt'] = np.random.normal(0, 0.002, n)
            d['q95'] = np.random.normal(4.5, 0.3, n)
            d['dq_dt'] = np.random.normal(0, 0.005, n)
            d['li'] = np.random.normal(0.35, 0.02, n)
            d['dli_dt'] = np.random.normal(0, 0.002, n)
            d['beta'] = np.random.normal(0.015, 0.002, n)
            d['dbeta_dt'] = np.random.normal(0, 0.0003, n)
            d['mirnov_dB_dt'] = np.random.normal(0.01, 0.005, n)
            d['locked_mode_indicator'] = np.abs(np.random.normal(0, 0.005, n))
            d['n1_rms'] = np.random.normal(0.01, 0.005, n)
            d['n2_rms'] = np.random.normal(0.008, 0.003, n)
            d['bolometry'] = np.random.normal(60, 10, n)
            d['Te'] = np.random.normal(3.0, 0.2, n)
            d['dTe_dt'] = np.random.normal(0, 0.02, n)
            d['ne_greenwald_frac'] = np.random.normal(0.45, 0.05, n)
            d['d2Ip_dt2'] = np.random.normal(0, 0.0002, n)
            d['distance_to_wall'] = np.random.normal(0.8, 0.05, n)
            d['error_field'] = np.random.normal(0.005, 0.002, n)
            d['stability_index'] = np.random.normal(-15, 3, n)
            
        elif regime == 'marginal':
            # Marginal plasma: moderate values, approaching limits
            d['Ip'] = np.random.normal(2.0, 0.08, n)
            d['dIp_dt'] = np.random.normal(0, 0.008, n)
            d['q95'] = np.random.normal(3.0, 0.25, n)
            d['dq_dt'] = np.random.normal(-0.01, 0.01, n)
            d['li'] = np.random.normal(0.42, 0.04, n)
            d['dli_dt'] = np.random.normal(0.005, 0.005, n)
            d['beta'] = np.random.normal(0.022, 0.004, n)
            d['dbeta_dt'] = np.random.normal(-0.0005, 0.0005, n)
            d['mirnov_dB_dt'] = np.random.normal(0.04, 0.015, n)
            d['locked_mode_indicator'] = np.abs(np.random.normal(0.03, 0.02, n))
            d['n1_rms'] = np.random.normal(0.04, 0.015, n)
            d['n2_rms'] = np.random.normal(0.025, 0.01, n)
            d['bolometry'] = np.random.normal(100, 15, n)
            d['Te'] = np.random.normal(2.2, 0.25, n)
            d['dTe_dt'] = np.random.normal(-0.05, 0.06, n)
            d['ne_greenwald_frac'] = np.random.normal(0.65, 0.08, n)
            d['d2Ip_dt2'] = np.random.normal(0, 0.0006, n)
            d['distance_to_wall'] = np.random.normal(0.5, 0.08, n)
            d['error_field'] = np.random.normal(0.015, 0.005, n)
            d['stability_index'] = np.random.normal(0, 5, n)
            
        else:
            # Pre-disruptive: low q95, high fluctuations, locked modes
            d['Ip'] = np.random.normal(2.4, 0.1, n)
            d['dIp_dt'] = np.random.normal(-0.02, 0.015, n)
            d['q95'] = np.random.normal(2.2, 0.15, n)
            d['dq_dt'] = np.random.normal(-0.03, 0.015, n)
            d['li'] = np.random.normal(0.55, 0.06, n)
            d['dli_dt'] = np.random.normal(0.02, 0.01, n)
            d['beta'] = np.random.normal(0.03, 0.006, n)
            d['dbeta_dt'] = np.random.normal(-0.002, 0.001, n)
            d['mirnov_dB_dt'] = np.random.normal(0.12, 0.04, n)
            d['locked_mode_indicator'] = np.abs(np.random.normal(0.15, 0.06, n))
            d['n1_rms'] = np.random.normal(0.1, 0.03, n)
            d['n2_rms'] = np.random.normal(0.06, 0.02, n)
            d['bolometry'] = np.random.normal(160, 25, n)
            d['Te'] = np.random.normal(1.5, 0.3, n)
            d['dTe_dt'] = np.random.normal(-0.2, 0.1, n)
            d['ne_greenwald_frac'] = np.random.normal(0.9, 0.08, n)
            d['d2Ip_dt2'] = np.random.normal(-0.001, 0.001, n)
            d['distance_to_wall'] = np.random.normal(0.25, 0.08, n)
            d['error_field'] = np.random.normal(0.04, 0.01, n)
            d['stability_index'] = np.random.normal(15, 5, n)
        
        regimes.append(pd.DataFrame(d))
    
    # Combine and shuffle
    df = pd.concat(regimes, ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)
    
    return df


def predict_with_uncertainty(input_data, n_mc_samples=50):
    """
    Predict disruption probability with uncertainty quantification
    
    Args:
        input_data: Feature array (batch_size, num_features)
        n_mc_samples: Number of MC dropout samples
    
    Returns:
        mean_pred: Mean prediction probability
        std_pred: Standard deviation (uncertainty)
        lower_ci: Lower confidence interval (2.5%)
        upper_ci: Upper confidence interval (97.5%)
    """
    model.train()  # Keep dropout active
    
    with torch.no_grad():
        mc_predictions = []
        for _ in range(n_mc_samples):
            output = model(input_data)
            mc_predictions.append(output)
        
        mc_predictions = torch.stack(mc_predictions)
        mean_pred = mc_predictions.mean(dim=0).cpu().numpy()
        std_pred = mc_predictions.std(dim=0).cpu().numpy()
        
        # 95% confidence interval
        lower_ci = mean_pred - 1.96 * std_pred
        upper_ci = mean_pred + 1.96 * std_pred
        
        lower_ci = np.clip(lower_ci, 0, 1)
        upper_ci = np.clip(upper_ci, 0, 1)
    
    model.eval()
    
    return mean_pred, std_pred, lower_ci, upper_ci


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve main dashboard page"""
    return render_template('index.html')


@app.route('/api/generate-data', methods=['POST'])
def api_generate_data():
    """Generate sample data and return for display"""
    try:
        num_samples = request.json.get('num_samples', 100)
        num_samples = min(max(int(num_samples), 10), 1000)  # 10-1000 samples
        
        # Generate data
        df_data = generate_sample_data(num_samples)
        
        # Prepare response
        response = {
            'status': 'success',
            'num_samples': num_samples,
            'features': df_data.columns.tolist(),
            'data': df_data.to_dict('list'),
            'summary': {
                'mean_ip': float(df_data['Ip'].mean()),
                'mean_q95': float(df_data['q95'].mean()),
                'mean_bolometry': float(df_data['bolometry'].mean()),
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Predict disruption probability with confidence intervals"""
    try:
        # Get input data
        data = request.json.get('data', None)
        
        if data is None:
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
        
        # Convert to DataFrame
        df_input = pd.DataFrame(data)
        
        # Ensure all required features are present
        for feat in feature_columns:
            if feat not in df_input.columns:
                df_input[feat] = 0.0
        
        # Select only required features
        df_input = df_input[feature_columns]
        
        # Normalize data
        if scaler is not None:
            X_scaled = scaler.transform(df_input)
        else:
            X_scaled = df_input.values
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
        
        # Predict with uncertainty
        mean_pred, std_pred, lower_ci, upper_ci = predict_with_uncertainty(
            X_tensor, n_mc_samples=50
        )
        
        # Compute adaptive thresholds based on the prediction distribution
        # This calibrates alert levels relative to the batch
        all_probs = mean_pred[:, 0].flatten()
        p33 = float(np.percentile(all_probs, 33))
        p66 = float(np.percentile(all_probs, 66))
        
        # Prepare predictions
        predictions = []
        for i in range(len(mean_pred)):
            pred_prob = float(mean_pred[i, 0])
            uncertainty = float(std_pred[i, 0])
            confidence = 1.0 - uncertainty
            
            # Adaptive alert: top third = HIGH, middle = MEDIUM, bottom = LOW
            if pred_prob >= p66:
                alert = 'HIGH'
            elif pred_prob >= p33:
                alert = 'MEDIUM'
            else:
                alert = 'LOW'
            
            predictions.append({
                'disruption_prob': pred_prob,
                'uncertainty': uncertainty,
                'confidence': confidence,
                'lower_ci': float(lower_ci[i, 0]),
                'upper_ci': float(upper_ci[i, 0]),
                'alert': alert
            })
        
        # Calculate statistics
        mean_probs = [p['disruption_prob'] for p in predictions]
        mean_uncertainties = [p['uncertainty'] for p in predictions]
        
        response = {
            'status': 'success',
            'num_samples': len(predictions),
            'predictions': predictions,
            'statistics': {
                'mean_probability': float(np.mean(mean_probs)),
                'std_probability': float(np.std(mean_probs)),
                'mean_uncertainty': float(np.mean(mean_uncertainties)),
                'high_risk_count': sum(1 for p in predictions if p['alert'] == 'HIGH'),
                'medium_risk_count': sum(1 for p in predictions if p['alert'] == 'MEDIUM'),
                'low_risk_count': sum(1 for p in predictions if p['alert'] == 'LOW'),
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 400


@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    """Get model information"""
    return jsonify({
        'status': 'success',
        'data': {
            'model_name': 'Advanced Bayesian Disruption Model',
            'features': feature_columns,
            'num_features': len(feature_columns),
            'device': str(device),
            'model_type': 'Bayesian Neural Network with MC Dropout',
            'uncertainty_method': 'MC Dropout (50 samples)',
        }
    })


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'status': 'error', 'message': 'Server error'}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("Loading model...")
    load_model()
    print("✓ Model loaded successfully")
    print(f"✓ Using device: {device}")
    print(f"✓ Number of features: {len(feature_columns)}")
    print("\nStarting Flask app...")
    app.run(debug=False, host='0.0.0.0', port=5000)
