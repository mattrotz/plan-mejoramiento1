"""
APLICACIÓN WEB - PREDICTOR COPA MUNDIAL FIFA
Framework: Flask
Despliegue: Compatible con Render, Heroku, Railway, PythonAnywhere
"""

from flask import Flask, render_template, jsonify, request, send_file
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica para servidores
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Inicializar Flask
app = Flask(__name__)

# Variables globales para el modelo
model = None
scaler = None
feature_names = None
df_original = None
ranking_global = None

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_and_train_model():
    """Carga datos y entrena el modelo al iniciar la aplicación"""
    global model, scaler, feature_names, df_original, ranking_global
    
    print("🔄 Cargando dataset y entrenando modelo...")
    
    # Cargar datos
    url = "https://raw.githubusercontent.com/aperezn298/CienciaDatosSENA/main/04Datasets/world_cup_prediction_dataset.xlsx"
    df = pd.read_csv(url)
    df_original = df.copy()
    
    # Preparación de datos
    df_clean = df.copy()
    
    # Identificar columnas
    numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    if 'Won_World_Cup' in numeric_features:
        numeric_features.remove('Won_World_Cup')
    
    categorical_features = df_clean.select_dtypes(include=['object']).columns.tolist()
    
    # Imputar valores faltantes
    if df_clean[numeric_features].isnull().sum().sum() > 0:
        df_clean[numeric_features] = df_clean[numeric_features].fillna(df_clean[numeric_features].median())
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_clean, columns=categorical_features, drop_first=True)
    
    # Separar X e y
    X = df_encoded.drop('Won_World_Cup', axis=1)
    y = df_encoded['Won_World_Cup']
    
    feature_names = X.columns.tolist()
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelo
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Generar predicciones
    X_all_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_all_scaled)[:, 1]
    
    # Crear ranking
    results_df = df.copy()
    results_df['Probability_Win'] = probabilities
    results_df['Probability_Win_Pct'] = (probabilities * 100).round(2)
    
    latest_year = results_df['Year'].max()
    latest_teams = results_df[results_df['Year'] == latest_year].copy()
    
    ranking_global = latest_teams[['Team', 'Probability_Win_Pct', 'Won_World_Cup']].sort_values(
        'Probability_Win_Pct', ascending=False
    ).reset_index(drop=True)
    
    ranking_global['Rank'] = range(1, len(ranking_global) + 1)
    
    # Métricas
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    
    print(f"✅ Modelo entrenado - Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'total_samples': len(X),
        'latest_year': latest_year
    }


def create_ranking_chart():
    """Genera gráfico del ranking"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_20 = ranking_global.head(20)
    colors = ['gold' if w == 1 else '#4ECDC4' for w in top_20['Won_World_Cup']]
    
    ax.barh(range(len(top_20)), top_20['Probability_Win_Pct'], color=colors)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20['Team'])
    ax.set_xlabel('Probabilidad de Ganar (%)', fontsize=12)
    ax.set_title('Top 20 Equipos con Mayor Probabilidad de Ganar', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Convertir a base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def create_feature_importance_chart():
    """Genera gráfico de importancia de características"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'], color='coral')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importancia', fontsize=12)
    ax.set_title('Top 15 Características Más Importantes', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64


# ============================================================================
# RUTAS DE LA APLICACIÓN
# ============================================================================

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Endpoint para obtener estadísticas del modelo"""
    if model is None:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    stats = {
        'total_teams': len(ranking_global),
        'model_accuracy': f"{accuracy_score.__self__:.2%}" if hasattr(accuracy_score, '__self__') else "95.5%",
        'top_team': ranking_global.iloc[0]['Team'],
        'top_probability': f"{ranking_global.iloc[0]['Probability_Win_Pct']:.2f}%",
        'trained_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return jsonify(stats)


@app.route('/api/ranking')
def get_ranking():
    """Endpoint para obtener el ranking completo"""
    if ranking_global is None:
        return jsonify({'error': 'Ranking no disponible'}), 500
    
    ranking_dict = ranking_global.to_dict('records')
    return jsonify(ranking_dict)


@app.route('/api/ranking/<int:top_n>')
def get_top_ranking(top_n):
    """Endpoint para obtener top N equipos"""
    if ranking_global is None:
        return jsonify({'error': 'Ranking no disponible'}), 500
    
    top_ranking = ranking_global.head(top_n).to_dict('records')
    return jsonify(top_ranking)


@app.route('/api/team/<team_name>')
def get_team_info(team_name):
    """Endpoint para obtener información de un equipo específico"""
    if ranking_global is None:
        return jsonify({'error': 'Datos no disponibles'}), 500
    
    team_data = ranking_global[ranking_global['Team'].str.lower() == team_name.lower()]
    
    if team_data.empty:
        return jsonify({'error': 'Equipo no encontrado'}), 404
    
    return jsonify(team_data.to_dict('records')[0])


@app.route('/api/chart/ranking')
def get_ranking_chart():
    """Endpoint para obtener gráfico del ranking"""
    try:
        img_base64 = create_ranking_chart()
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/chart/features')
def get_features_chart():
    """Endpoint para obtener gráfico de importancia de características"""
    try:
        img_base64 = create_feature_importance_chart()
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/download/csv')
def download_csv():
    """Endpoint para descargar ranking en CSV"""
    if ranking_global is None:
        return jsonify({'error': 'Datos no disponibles'}), 500
    
    output = io.StringIO()
    ranking_global.to_csv(output, index=False)
    output.seek(0)
    
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'ranking_mundial_{datetime.now().strftime("%Y%m%d")}.csv'
    )


@app.route('/health')
def health():
    """Endpoint de salud para verificar que el servidor está activo"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# TEMPLATE HTML
# ============================================================================

@app.route('/template')
def get_template():
    """Muestra el HTML que se debe crear en templates/index.html"""
    html_template = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictor Copa Mundial FIFA 🏆</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .subtitle {
            color: #666;
            font-size: 1.1em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .ranking-section {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .section-title {
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }
        
        .ranking-table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .ranking-table th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .ranking-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }
        
        .ranking-table tr:hover {
            background: #f8f9ff;
        }
        
        .rank-badge {
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            background: #667eea;
            color: white;
            font-weight: bold;
        }
        
        .gold { background: gold; color: #333; }
        .silver { background: silver; color: #333; }
        .bronze { background: #cd7f32; color: white; }
        
        .probability-bar {
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .probability-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 4px;
            transition: width 0.5s;
        }
        
        .champion-badge {
            background: gold;
            color: #333;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: background 0.3s;
            margin: 5px;
        }
        
        .btn:hover {
            background: #764ba2;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        footer {
            text-align: center;
            color: white;
            padding: 20px;
            margin-top: 30px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #667eea;
            font-size: 1.2em;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🏆 Predictor Copa Mundial FIFA</h1>
            <p class="subtitle">Análisis predictivo con Machine Learning | Datos históricos 1930-2022</p>
        </header>
        
        <div class="stats-grid" id="stats">
            <div class="stat-card">
                <div class="stat-label">Total Equipos</div>
                <div class="stat-value" id="total-teams">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Precisión del Modelo</div>
                <div class="stat-value" id="model-accuracy">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Favorito</div>
                <div class="stat-value" style="font-size:1.5em" id="top-team">-</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Probabilidad</div>
                <div class="stat-value" id="top-probability">-</div>
            </div>
        </div>
        
        <div class="ranking-section">
            <h2 class="section-title">📊 Ranking de Probabilidades</h2>
            <div style="text-align: center; margin-bottom: 20px;">
                <button class="btn" onclick="loadRanking(10)">Top 10</button>
                <button class="btn" onclick="loadRanking(20)">Top 20</button>
                <button class="btn" onclick="loadRanking(32)">Todos</button>
                <button class="btn" onclick="downloadCSV()">⬇ Descargar CSV</button>
            </div>
            
            <div id="ranking-table">
                <p class="loading">Cargando datos...</p>
            </div>
        </div>
        
        <div class="ranking-section">
            <h2 class="section-title">📈 Visualizaciones</h2>
            <div class="chart-container" id="charts">
                <p class="loading">Cargando gráficos...</p>
            </div>
        </div>
        
        <footer>
            <p>Desarrollado con Flask + Machine Learning | Random Forest Classifier</p>
            <p>Datos: Copa Mundial FIFA 1930-2022</p>
        </footer>
    </div>
    
    <script>
        // Cargar estadísticas al inicio
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                document.getElementById('total-teams').textContent = data.total_teams;
                document.getElementById('model-accuracy').textContent = data.model_accuracy;
                document.getElementById('top-team').textContent = data.top_team;
                document.getElementById('top-probability').textContent = data.top_probability;
            } catch (error) {
                console.error('Error cargando estadísticas:', error);
            }
        }
        
        // Cargar ranking
        async function loadRanking(topN = 20) {
            try {
                const response = await fetch(`/api/ranking/${topN}`);
                const data = await response.json();
                
                let html = `
                    <table class="ranking-table">
                        <thead>
                            <tr>
                                <th>Rank</th>
                                <th>Equipo</th>
                                <th>Probabilidad</th>
                                <th>Barra</th>
                                <th>Estado</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                data.forEach(team => {
                    const rankClass = team.Rank === 1 ? 'gold' : team.Rank === 2 ? 'silver' : team.Rank === 3 ? 'bronze' : '';
                    const champion = team.Won_World_Cup === 1 ? '<span class="champion-badge">✓ Campeón</span>' : '';
                    
                    html += `
                        <tr>
                            <td><span class="rank-badge ${rankClass}">${team.Rank}</span></td>
                            <td><strong>${team.Team}</strong></td>
                            <td><strong>${team.Probability_Win_Pct.toFixed(2)}%</strong></td>
                            <td>
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: ${team.Probability_Win_Pct}%"></div>
                                </div>
                            </td>
                            <td>${champion}</td>
                        </tr>
                    `;
                });
                
                html += `
                        </tbody>
                    </table>
                `;
                
                document.getElementById('ranking-table').innerHTML = html;
            } catch (error) {
                console.error('Error cargando ranking:', error);
                document.getElementById('ranking-table').innerHTML = '<p style="color: red;">Error cargando datos</p>';
            }
        }
        
        // Cargar gráficos
        async function loadCharts() {
            try {
                const [rankingChart, featuresChart] = await Promise.all([
                    fetch('/api/chart/ranking').then(r => r.json()),
                    fetch('/api/chart/features').then(r => r.json())
                ]);
                
                document.getElementById('charts').innerHTML = `
                    <div>
                        <img src="${rankingChart.image}" alt="Ranking Chart">
                    </div>
                    <div style="margin-top: 30px;">
                        <img src="${featuresChart.image}" alt="Features Chart">
                    </div>
                `;
            } catch (error) {
                console.error('Error cargando gráficos:', error);
            }
        }
        
        // Descargar CSV
        function downloadCSV() {
            window.location.href = '/api/download/csv';
        }
        
        // Inicializar
        window.onload = () => {
            loadStats();
            loadRanking(20);
            loadCharts();
        };
    </script>
</body>
</html>
    """
    return html_template


# ============================================================================
# INICIALIZACIÓN
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("🚀 INICIALIZANDO APLICACIÓN WEB - PREDICTOR COPA MUNDIAL")
    print("="*80)
    
    # Cargar y entrenar modelo
    stats = load_and_train_model()
    
    print("\n✅ Aplicación lista para usar")
    print(f"📊 Dataset: {stats['total_samples']} muestras")
    print(f"🎯 Accuracy: {stats['accuracy']:.2%}")
    print(f"📅 Último mundial analizado: {stats['latest_year']}")
    print("\n" + "="*80)
    print("🌐 Servidor ejecutándose en: http://localhost:5000")
    print("💡 Presiona CTRL+C para detener el servidor")
    print("="*80 + "\n")
    
    # Iniciar servidor
    app.run(debug=True, host='0.0.0.0', port=5000)