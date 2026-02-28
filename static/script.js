/* ============================================================================
   PLASMA DISRUPTION DASHBOARD - JAVASCRIPT
   ============================================================================ */

// Global state management
const state = {
    generatedData: null,
    predictions: null,
    charts: {},
    isLoading: false
};

// Chart.js global options (v4 API)
Chart.defaults.responsive = true;
Chart.defaults.maintainAspectRatio = true;
Chart.defaults.animation = { duration: 500 };

/* ============================================================================
   INITIALIZATION
   ============================================================================ */

document.addEventListener('DOMContentLoaded', function() {
    console.log('Dashboard initialized');
    
    // Load model info
    loadModelInfo();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize empty charts
    initializeCharts();
});

function setupEventListeners() {
    const generateBtn = document.getElementById('generate-btn');
    const predictBtn = document.getElementById('predict-btn');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateData);
    }
    
    if (predictBtn) {
        predictBtn.addEventListener('click', predict);
    }
}

/* ============================================================================
   MODEL INFO LOADING
   ============================================================================ */

async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        if (data.status === 'success') {
            displayModelInfo(data.data);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
        showStatusMessage('generate-status', 'Error loading model info', 'error');
    }
}

function displayModelInfo(info) {
    const modelInfoDiv = document.getElementById('model-info');
    if (!modelInfoDiv) return;
    
    let html = '<div class="model-info-item">';
    html += '<span class="model-info-label">Model:</span>';
    html += '<span class="model-info-value">' + info.model_name + '</span>';
    html += '</div>';
    
    html += '<div class="model-info-item">';
    html += '<span class="model-info-label">Features:</span>';
    html += '<span class="model-info-value">' + info.num_features + '</span>';
    html += '</div>';
    
    html += '<div class="model-info-item">';
    html += '<span class="model-info-label">Device:</span>';
    html += '<span class="model-info-value">' + info.device + '</span>';
    html += '</div>';
    
    html += '<div class="model-info-item">';
    html += '<span class="model-info-label">UQ Method:</span>';
    html += '<span class="model-info-value">' + info.uncertainty_method + '</span>';
    html += '</div>';
    
    modelInfoDiv.innerHTML = html;
}

/* ============================================================================
   DATA GENERATION
   ============================================================================ */

async function generateData() {
    try {
        // Get number of samples
        const numSamplesInput = document.getElementById('num-samples');
        const numSamples = parseInt(numSamplesInput.value) || 100;
        
        if (numSamples < 10 || numSamples > 1000) {
            showStatusMessage('generate-status', 'Samples must be between 10 and 1000', 'error');
            return;
        }
        
        // Show loading state
        state.isLoading = true;
        showStatusMessage('generate-status', 'Generating data...', 'loading');
        document.getElementById('generate-btn').disabled = true;
        document.getElementById('predict-btn').disabled = true;
        
        // Send request
        const response = await fetch('/api/generate-data', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_samples: numSamples })
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            state.generatedData = result.data;
            showStatusMessage('generate-status', 
                'Generated ' + numSamples + ' samples successfully!', 'success');
            document.getElementById('predict-btn').disabled = false;
            displayFeatureShowcase(result.data);
        } else {
            showStatusMessage('generate-status', result.message || 'Error generating data', 'error');
        }
    } catch (error) {
        console.error('Error generating data:', error);
        showStatusMessage('generate-status', 'Error: ' + error.message, 'error');
    } finally {
        state.isLoading = false;
        document.getElementById('generate-btn').disabled = false;
    }
}

/* ============================================================================
   PREDICTION
   ============================================================================ */

async function predict() {
    try {
        if (!state.generatedData) {
            showStatusMessage('predict-status', 'Please generate data first', 'error');
            return;
        }
        
        // Show loading state
        state.isLoading = true;
        showStatusMessage('predict-status', 'Making predictions...', 'loading');
        document.getElementById('predict-btn').disabled = true;
        
        // Prepare data
        const payload = {
            data: state.generatedData
        };
        
        // Send request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (result.status === 'success') {
            state.predictions = result;
            showStatusMessage('predict-status', 'Predictions complete!', 'success');
            
            // Update all visualizations
            updateStatistics(state.predictions);
            updateCharts(state.predictions);
            displaySampleCIChart(state.predictions);
            populateTable(state.predictions);
        } else {
            showStatusMessage('predict-status', result.message || 'Error making predictions', 'error');
        }
    } catch (error) {
        console.error('Error making predictions:', error);
        showStatusMessage('predict-status', 'Error: ' + error.message, 'error');
    } finally {
        state.isLoading = false;
        document.getElementById('predict-btn').disabled = false;
    }
}

/* ============================================================================
   STATISTICS UPDATE
   ============================================================================ */

function updateStatistics(predictions) {
    const stats = predictions.statistics;
    
    // Mean Probability
    const meanProbValue = document.querySelector('[data-stat="mean-prob"] .stat-value');
    if (meanProbValue) {
        meanProbValue.textContent = (stats.mean_probability * 100).toFixed(1) + '%';
    }
    
    // Mean Uncertainty
    const meanUncValue = document.querySelector('[data-stat="mean-uncertainty"] .stat-value');
    if (meanUncValue) {
        meanUncValue.textContent = (stats.mean_uncertainty * 100).toFixed(1) + '%';
    }
    
    // High Risk Count
    const highRiskValue = document.querySelector('[data-stat="high-risk"] .stat-value');
    if (highRiskValue) {
        highRiskValue.textContent = stats.high_risk_count;
        highRiskValue.className = 'stat-value stat-high';
    }
    
    // Medium Risk Count
    const mediumRiskValue = document.querySelector('[data-stat="medium-risk"] .stat-value');
    if (mediumRiskValue) {
        mediumRiskValue.textContent = stats.medium_risk_count;
        mediumRiskValue.className = 'stat-value stat-medium';
    }
}

/* ============================================================================
   CHART INITIALIZATION
   ============================================================================ */

function initializeCharts() {
    // Create empty chart contexts
    const probabilityCtx = document.getElementById('probability-chart');
    const riskCtx = document.getElementById('risk-chart');
    const scatterCtx = document.getElementById('scatter-chart');
    
    if (probabilityCtx) {
        state.charts.probability = new Chart(probabilityCtx, {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Disruption Probability Distribution',
                    data: [],
                    backgroundColor: 'rgba(46, 134, 171, 0.6)',
                    borderColor: 'rgba(46, 134, 171, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }
    
    if (riskCtx) {
        state.charts.risk = new Chart(riskCtx, {
            type: 'pie',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: []
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true
            }
        });
    }
    
    if (scatterCtx) {
        state.charts.scatter = new Chart(scatterCtx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Probability vs Uncertainty',
                    data: [],
                    backgroundColor: 'rgba(162, 59, 114, 0.6)',
                    borderColor: 'rgba(162, 59, 114, 1)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    x: { 
                        title: { display: true, text: 'Probability' },
                        min: 0, max: 1
                    },
                    y: { 
                        title: { display: true, text: 'Uncertainty' },
                        min: 0, max: 1
                    }
                }
            }
        });
    }
}

/* ============================================================================
   CHART UPDATES
   ============================================================================ */

function updateCharts(predictions) {
    updateProbabilityChart(predictions);
    updateRiskChart(predictions);
    updateScatterChart(predictions);
    updateConfidenceChart(predictions);
}

function updateProbabilityChart(predictions) {
    if (!state.charts.probability) return;
    
    const probs = predictions.predictions.map(p => p.disruption_prob);
    
    // Create histogram bins
    const bins = 10;
    const binWidth = 1 / bins;
    const histogram = Array(bins).fill(0);
    
    probs.forEach(prob => {
        const binIndex = Math.min(Math.floor(prob / binWidth), bins - 1);
        histogram[binIndex]++;
    });
    
    const labels = [];
    for (let i = 0; i < bins; i++) {
        labels.push((i * binWidth).toFixed(2) + '-' + ((i + 1) * binWidth).toFixed(2));
    }
    
    state.charts.probability.data.labels = labels;
    state.charts.probability.data.datasets[0].data = histogram;
    state.charts.probability.update();
}

function updateRiskChart(predictions) {
    if (!state.charts.risk) return;
    
    const stats = predictions.statistics;
    const total = stats.high_risk_count + stats.medium_risk_count + 
                  (predictions.predictions.length - stats.high_risk_count - stats.medium_risk_count);
    
    state.charts.risk.data.labels = ['High Risk', 'Medium Risk', 'Low Risk'];
    state.charts.risk.data.datasets[0].data = [
        stats.high_risk_count,
        stats.medium_risk_count,
        total - stats.high_risk_count - stats.medium_risk_count
    ];
    state.charts.risk.data.datasets[0].backgroundColor = [
        'rgba(242, 66, 54, 0.6)',
        'rgba(241, 143, 1, 0.6)',
        'rgba(6, 167, 125, 0.6)'
    ];
    state.charts.risk.update();
}

function updateScatterChart(predictions) {
    if (!state.charts.scatter) return;
    
    const data = predictions.predictions.map(p => ({
        x: p.disruption_prob,
        y: p.upper_ci - p.lower_ci
    }));
    
    state.charts.scatter.data.datasets[0].data = data;
    state.charts.scatter.update();
}

function updateConfidenceChart(predictions) {
    const ciDiv = document.getElementById('confidence-chart');
    if (!ciDiv) return;
    
    // Clear existing plotly chart
    Plotly.purge('confidence-chart');
    
    // Prepare data for Plotly
    const x = predictions.predictions.map((_, i) => i);
    const prob = predictions.predictions.map(p => p.disruption_prob);
    const lower = predictions.predictions.map(p => p.lower_ci);
    const upper = predictions.predictions.map(p => p.upper_ci);
    
    // Main line
    const trace1 = {
        x: x,
        y: prob,
        name: 'Predicted Probability',
        mode: 'lines',
        fill: 'none',
        line: { color: '#2e86ab', width: 2 }
    };
    
    // Upper CI band
    const trace2 = {
        x: x,
        y: upper,
        name: 'Upper CI (95%)',
        mode: 'lines',
        line: { color: 'rgba(0,0,0,0)' },
        showlegend: false
    };
    
    // Lower CI band
    const trace3 = {
        x: x,
        y: lower,
        name: 'Lower CI (95%)',
        mode: 'lines',
        fill: 'tonexty',
        line: { color: 'rgba(0,0,0,0)' },
        fillcolor: 'rgba(46, 134, 171, 0.2)',
        showlegend: false
    };
    
    const isMobile = window.innerWidth < 768;
    const layout = {
        title: isMobile ? '' : 'Confidence Intervals (95%)',
        xaxis: { title: isMobile ? '' : 'Sample Index', automargin: true },
        yaxis: { title: isMobile ? 'P' : 'Disruption Probability', range: [-0.1, 1.1], automargin: true },
        hovermode: 'closest',
        autosize: true,
        margin: isMobile ? { l: 35, r: 10, t: 10, b: 35 } : { l: 50, r: 50, t: 50, b: 50 }
    };
    
    Plotly.newPlot('confidence-chart', [trace2, trace3, trace1], layout, { responsive: true });
}

/* ============================================================================
   FEATURE SHOWCASE
   ============================================================================ */

const FEATURE_META = {
    'Ip':                     { desc: 'Plasma Current',                  unit: 'MA',    category: 'current' },
    'dIp_dt':                 { desc: 'Current Ramp Rate',               unit: 'MA/s',  category: 'current' },
    'd2Ip_dt2':               { desc: 'Current Acceleration',            unit: 'MA/s²', category: 'current' },
    'q95':                    { desc: 'Safety Factor (edge)',             unit: '',      category: 'safety' },
    'dq_dt':                  { desc: 'Safety Factor Rate',              unit: '1/s',   category: 'safety' },
    'li':                     { desc: 'Internal Inductance',             unit: '',      category: 'stability' },
    'dli_dt':                 { desc: 'Inductance Rate',                 unit: '1/s',   category: 'stability' },
    'beta':                   { desc: 'Normalized Beta',                 unit: '',      category: 'stability' },
    'dbeta_dt':               { desc: 'Beta Rate of Change',             unit: '1/s',   category: 'stability' },
    'mirnov_dB_dt':           { desc: 'Mirnov Coil dB/dt',              unit: 'T/s',   category: 'mhd' },
    'locked_mode_indicator':  { desc: 'Locked Mode Amplitude',           unit: 'a.u.',  category: 'mhd' },
    'n1_rms':                 { desc: 'n=1 Mode RMS Amplitude',          unit: 'a.u.',  category: 'mhd' },
    'n2_rms':                 { desc: 'n=2 Mode RMS Amplitude',          unit: 'a.u.',  category: 'mhd' },
    'bolometry':              { desc: 'Radiated Power',                  unit: 'kW',    category: 'kinetic' },
    'Te':                     { desc: 'Electron Temperature',            unit: 'keV',   category: 'kinetic' },
    'dTe_dt':                 { desc: 'Temperature Rate',                unit: 'keV/s', category: 'kinetic' },
    'ne_greenwald_frac':      { desc: 'Greenwald Density Fraction',      unit: '',      category: 'kinetic' },
    'distance_to_wall':       { desc: 'Plasma-Wall Gap',                 unit: 'm',     category: 'stability' },
    'error_field':            { desc: 'Error Field Magnitude',           unit: 'a.u.',  category: 'mhd' },
    'stability_index':        { desc: 'Composite Stability Index',       unit: '',      category: 'stability' },
};

function displayFeatureShowcase(generatedData) {
    const container = document.getElementById('features-showcase');
    const grid = document.getElementById('features-grid');
    if (!container || !grid) return;

    const features = Object.keys(generatedData).filter(k => FEATURE_META[k]);
    if (features.length === 0) return;

    grid.innerHTML = '';
    features.forEach(feat => {
        const meta = FEATURE_META[feat] || { desc: feat, unit: '', category: 'stability' };
        const values = generatedData[feat];
        const mean = (values.reduce((a, b) => a + b, 0) / values.length);
        const min = Math.min(...values);
        const max = Math.max(...values);

        const card = document.createElement('div');
        card.className = 'feature-card category-' + meta.category;
        card.innerHTML =
            '<div class="feature-name">' + feat + '</div>' +
            '<div class="feature-desc">' + meta.desc + (meta.unit ? ' (' + meta.unit + ')' : '') + '</div>' +
            '<div class="feature-stat">μ=' + mean.toFixed(4) + '  [' + min.toFixed(3) + ', ' + max.toFixed(3) + ']</div>';
        grid.appendChild(card);
    });

    container.style.display = 'block';
}

/* ============================================================================
   PER-SAMPLE CONFIDENCE INTERVAL CHART
   ============================================================================ */

function displaySampleCIChart(predictions) {
    const box = document.getElementById('sample-ci-box');
    const div = document.getElementById('sample-ci-chart');
    if (!box || !div) return;

    Plotly.purge('sample-ci-chart');

    // Take first 20 samples
    const preds = predictions.predictions.slice(0, 20);
    const x = preds.map((_, i) => 'S' + (i + 1));
    const y = preds.map(p => p.disruption_prob);
    const errUp = preds.map(p => p.upper_ci - p.disruption_prob);
    const errDn = preds.map(p => p.disruption_prob - p.lower_ci);
    const colors = preds.map(p =>
        p.alert === 'HIGH' ? '#f24236' : p.alert === 'MEDIUM' ? '#f18f01' : '#06a77d'
    );

    const trace = {
        x: x,
        y: y,
        error_y: {
            type: 'data',
            symmetric: false,
            array: errUp,
            arrayminus: errDn,
            color: '#555',
            thickness: 1.5,
            width: 6,
        },
        type: 'bar',
        marker: { color: colors, opacity: 0.85 },
        text: preds.map(p => p.alert),
        hovertemplate: '<b>%{x}</b><br>P = %{y:.3f}<br>CI: [%{customdata[0]:.3f}, %{customdata[1]:.3f}]<br>Alert: %{text}<extra></extra>',
        customdata: preds.map(p => [p.lower_ci, p.upper_ci]),
    };

    const isMobile = window.innerWidth < 768;
    const layout = {
        yaxis: { title: isMobile ? 'P' : 'Disruption Probability', range: [0, 1.15], automargin: true },
        xaxis: { title: isMobile ? '' : 'Sample', automargin: true },
        autosize: true,
        margin: isMobile ? { l: 35, r: 10, t: 10, b: 35 } : { l: 55, r: 20, t: 20, b: 50 },
        bargap: 0.3,
        shapes: [
            { type: 'line', x0: -0.5, x1: 19.5, y0: 0.5, y1: 0.5,
              line: { color: '#aaa', width: 1, dash: 'dash' } }
        ],
        annotations: [
            { x: 19.5, y: 0.52, text: 'P = 0.5', showarrow: false,
              font: { size: 10, color: '#888' }, xanchor: 'right' }
        ]
    };

    Plotly.newPlot('sample-ci-chart', [trace], layout, { responsive: true });
    box.style.display = 'block';
}

/* ============================================================================
   TABLE POPULATION
   ============================================================================ */

function populateTable(predictions) {
    const tableBody = document.getElementById('predictions-table-body');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    predictions.predictions.forEach((pred, index) => {
        const row = document.createElement('tr');
        
        // Index
        const indexCell = document.createElement('td');
        indexCell.textContent = index + 1;
        row.appendChild(indexCell);
        
        // Probability
        const probCell = document.createElement('td');
        probCell.textContent = (pred.disruption_prob * 100).toFixed(2) + '%';
        row.appendChild(probCell);
        
        // Confidence (Uncertainty)
        const confCell = document.createElement('td');
        confCell.textContent = (pred.confidence * 100).toFixed(2) + '%';
        row.appendChild(confCell);
        
        // 95% CI
        const ciCell = document.createElement('td');
        ciCell.textContent = '[' + (pred.lower_ci * 100).toFixed(2) + '%, ' + (pred.upper_ci * 100).toFixed(2) + '%]';
        row.appendChild(ciCell);
        
        // Alert Level
        const alertCell = document.createElement('td');
        const alertClass = 'alert-badge ' + pred.alert.toLowerCase();
        alertCell.innerHTML = '<span class="' + alertClass + '">' + pred.alert + '</span>';
        row.appendChild(alertCell);
        
        tableBody.appendChild(row);
    });
}

/* ============================================================================
   UI HELPERS
   ============================================================================ */

function showStatusMessage(elementId, message, type) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    element.textContent = message;
    element.className = 'status-message ' + type;
    
    if (type === 'success' || type === 'error') {
        setTimeout(() => {
            element.className = 'status-message';
            element.textContent = '';
        }, 3000);
    }
}

/* ============================================================================
   ERROR HANDLING
   ============================================================================ */

window.addEventListener('error', function(event) {
    console.error('Global error:', event.error);
    showStatusMessage('predict-status', 'An unexpected error occurred', 'error');
});
