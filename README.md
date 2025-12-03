# HBA1C Prediction MLOps Project

Production-grade healthcare data analytics and machine learning project for predicting HBA1C levels from blood sugar values.

## 🎯 Project Overview

This project predicts HBA1C levels (a crucial marker for diabetes management) using blood sugar readings. By leveraging machine learning with comprehensive MLOps practices, the system provides accurate HBA1C predictions to assist in diabetes monitoring and management.

### Key Features
- **Real-time Predictions**: Interactive web interface for patient HBA1C prediction
- **Comprehensive Analytics**: In-depth exploratory data analysis and visualization
- **Production ML Pipeline**: End-to-end MLOps with model training, tracking, and deployment
- **Dynamic Model Loading**: Models fetched directly from DagsHub repository

## 📊 Data Analytics

### Jupyter Notebooks Analysis
The project includes comprehensive data analysis performed in the `Data Analysis Python Pandas` directory:

- **Healthcare Data Analytics Python Pandas.ipynb**: Full exploratory data analysis including:
  - Blood sugar level distributions and outlier detection
  - HBA1C level analysis with box plots and histograms
  - Blood pressure correlations and health metrics
  - Correlation heatmap between health parameters
  - Patient demographic analysis

- **Healthcare ML Model Without MLOPS.ipynb**: Initial model development and validation

### Key Insights Discovered
- Strong positive correlation between blood sugar levels and HBA1C
- Seasonal variations in blood sugar readings
- Patient-specific patterns and trends
- Relationship between blood pressure and diabetes indicators

## 🤖 Machine Learning Model

### Model Architecture
- **Algorithm**: RandomForestRegressor
- **Parameters**: 100 estimators with optimized hyperparameters
- **Training Dataset**: 14,000+ synthetic patient records
- **Features Used**:
  - Patient ID
  - Blood Sugar Level
  - Date features (Year, Month, Day)
  - Historical health metrics

### Model Performance
- Robust regression model with high accuracy
- Handles non-linear relationships effectively
- Feature importance analysis reveals key predictive factors

## 🛠️ MLOps Stack

### Workflow Orchestration
- **[Metaflow]**: Data science workflow orchestration
  - Handles end-to-end ML pipeline
  - Versioned steps and reproducible runs
  - Automatic branching and parallel processing

### Experiment Tracking
- **[MLflow]**: Comprehensive ML lifecycle management
  - Experiment tracking and parameter logging
  - Model versioning and artifact storage
  - Performance metrics visualization
  - Model registry management

### Model Repository & Version Control
- **[DagsHub]**: ML collaboration platform
  - Git-based model versioning
  - MLflow tracking server integration
  - Model artifact storage and retrieval
  - Collaboration features for ML teams

### Model Monitoring
- **[Evidently AI]**: ML monitoring and drift detection
  - Data drift analysis
  - Performance monitoring
  - Model quality dashboards

## 🚀 Deployment

### Production Deployment
- **[Railway]**: Cloud deployment platform
  - Automatic CI/CD from GitHub
  - Environment variable management
  - Scalable web application hosting
  - Production-ready configuration with Gunicorn

### Application Architecture
- **Backend**: Flask web application
- **Model Loading**: Dynamic fetching from DagsHub MLflow
- **Frontend**: Bootstrap-based responsive UI
- **API**: RESTful endpoints for predictions

## 📋 Project Structure

```
hba1cmetaflow/
├── Data/                           # Dataset files
├── Data Analysis Python Pandas/    # Jupyter notebooks for EDA
├── templates/                      # HTML templates for web app
├── .metaflow/                     # Metaflow run artifacts
├── mlruns/                        # MLflow local tracking
├── app.py                         # Flask web application
├── regression_flow.py            # Metaflow training pipeline
├── mlops_mlflow_metaflow.py      # Enhanced MLOps pipeline
├── requirements.txt               # Python dependencies
├── Procfile                       # Railway deployment config
└── .env                          # Environment variables (local)
```

## 🏃‍♂️ Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/sakthi-t/hba1cmetaflow.git
   cd hba1cmetaflow
   ```

2. **Set up environment**
   ```bash
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   # Create .env file
   echo "DAGSHUB_TOKEN=your_dagshub_token" > .env
   ```

4. **Run the web application**
   ```bash
   python3 app.py
   ```
   Visit `http://localhost:5000` to access the application

### Training New Models

1. **Run the Metaflow pipeline**
   ```bash
   python3 mlops_mlflow_metaflow.py run
   ```

2. **Update model run ID in app.py** (after successful training)

### Deploy to Railway

1. **Push to GitHub** (Railway auto-deploys)
   ```bash
   git add .
   git commit -m "Update deployment"
   git push origin main
   ```

2. **Configure Railway**
   - Set `DAGSHUB_TOKEN` in environment variables
   - Railway will automatically deploy from the Procfile

## 🌐 Live Application

**Deployed Application**: [https://web-production-5c6f8.up.railway.app/](https://web-production-5c6f8.up.railway.app/)

The live application demonstrates:
- Real-time HBA1C predictions
- Model loading from remote MLflow server
- Production-ready web interface
- Dynamic model version management

## 📈 Model Registry

Models are tracked and versioned at:
- **DagsHub MLflow**: [https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow](https://dagshub.com/sakthi-t/hba1cmetaflow.mlflow)
- **Latest Run**: fb8d1d9483ef436baee58405144978e4

## 🧪 Testing

### Run Tests
```bash
# Test model loading
python3 -c "from app import load_model; load_model()"

# Test prediction
python3 -c "from app import predict_hba1c; print(predict_hba1c('P001', '2024-01-15', 150))"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project uses synthetic data for demonstration purposes. The predictions should not be used for actual medical diagnosis. Always consult healthcare professionals for medical advice.

## 🔗 Links

- **GitHub Repository**: [https://github.com/sakthi-t/hba1cmetaflow](https://github.com/sakthi-t/hba1cmetaflow)
- **DagsHub Repository**: [https://dagshub.com/sakthi-t/hba1cmetaflow](https://dagshub.com/sakthi-t/hba1cmetaflow)
- **Live Application**: [https://web-production-5c6f8.up.railway.app/](https://web-production-5c6f8.up.railway.app/)

---

Built with ❤️ using Metaflow, MLflow, DagsHub, and Railway