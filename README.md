# AI-Enhanced Blood Pressure Monitoring System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-red)

A comprehensive machine learning system for cuffless blood pressure estimation and cardiovascular disease (CVD) risk assessment using physiological signal processing. This project was developed as part of a Master's Group Design Project at the University of Southampton (2024-2025).

## 🎓 Project Overview

This MEng Group Design Project investigates data processing components for a wearable blood pressure monitoring system. The project focuses on two key areas:
1. **CVD Risk Assessment**: Machine learning model achieving 78.66% accuracy
2. **Cuffless Blood Pressure Estimation**: Achieving R² scores of 0.824 (systolic) and 0.733 (diastolic)

The system processes pressure pulse wave signals from piezoresistive sensors to provide continuous, non-invasive blood pressure monitoring enhanced by artificial intelligence.

### Key Features

- **Dual ML Systems**: CVD risk prediction and blood pressure estimation
- **Advanced Signal Processing**: Butterworth bandpass filtering (0.5-3 Hz) with zero-phase distortion
- **Clinical-Grade Performance**: Mean errors below 5 mmHg for blood pressure estimation
- **Robust Preprocessing**: Comprehensive noise simulation and reduction techniques
- **SMOTE Implementation**: Addresses class imbalance in cardiovascular datasets
- **Multi-Model Ensemble**: Random Forest, XGBoost, and Gradient Boosting

## 📊 Results Summary

### Blood Pressure Estimation Performance

| Metric | Systolic BP | Diastolic BP |
|--------|-------------|---------------|
| **R² Score** | 0.824 | 0.733 |
| **Mean Error** | 2.300 ± 0.076 mmHg | 1.391 ± 0.034 mmHg |
| **Within 5 mmHg** | >85% | >85% |

### CVD Risk Assessment Performance

| Model | Accuracy |
|-------|----------|
| **Random Forest** | 78.77% |
| **XGBoost** | 78.66% |
| **Gradient Boosting** | 73.70% |

## 🛠️ Technical Architecture

### 1. Blood Pressure Estimation System

#### Signal Processing Pipeline
- **Noise Simulation**: Gaussian (σ=0.015), Uniform (0.001-0.015V), Mains interference (50 Hz, 0.005V)
- **Filtering**: Butterworth bandpass filter (0.5-3 Hz) with forward-backward technique
- **Feature Extraction**: 21 morphological features from pulse waveforms
  - Systolic/diastolic peaks and timing
  - Dicrotic notch characteristics
  - Pulse width and area under curve
  - Rise/fall time ratios

#### Machine Learning Models
- **Random Forest Regressor**: Primary model for BP estimation
- **Gradient Boosting Regression**: Enhanced performance on medical datasets
- **AdaBoost Regression**: Robustness to signal variations

### 2. CVD Risk Assessment System

#### Data Processing
- **Dataset**: Framingham Heart Study (4,238 participants)
- **Class Balancing**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Engineering**: Blood pressure, BMI, heart rate, smoking status, lipid profiles

#### Classification Models
- **Random Forest Classifier** (78.77% accuracy)
- **XGBoost** (78.66% accuracy) 
- **Gradient Boosting** (73.70% accuracy)

## 💾 Datasets

### MIMIC-III Waveform Database
- **Source**: PhysioNet
- **Size**: 40,000+ ICU patients
- **Usage**: Blood pressure estimation
- **Preprocessing**: High-fidelity ABP waveform extraction and quality filtering

### Framingham Heart Study
- **Source**: Kaggle
- **Size**: 4,238 participants
- **Usage**: CVD risk assessment
- **Features**: 10-year cardiovascular event tracking with comprehensive risk factors

## ⚙️ Installation

### Prerequisites
```
Python 3.8+
NumPy
Pandas
Scikit-learn
SciPy
Matplotlib
Seaborn
imbalanced-learn (for SMOTE)
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/mukilan2101/AI-Enhanced-Blood-Pressure-Monitoring.git
cd AI-Enhanced-Blood-Pressure-Monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
- MIMIC-III: Register at PhysioNet and complete required training
- Framingham: Available on Kaggle

## 🚀 Usage

### Blood Pressure Estimation

```python
from bp_estimation import WaveformProcessor, BPPredictor

# Initialize processor
processor = WaveformProcessor()

# Load and process waveform
waveform = processor.load_waveform('path/to/data')
processed = processor.apply_filtering(waveform)
features = processor.extract_features(processed)

# Predict blood pressure
predictor = BPPredictor()
sbp, dbp = predictor.predict(features)
print(f"Systolic: {sbp:.1f} mmHg, Diastolic: {dbp:.1f} mmHg")
```

### CVD Risk Assessment

```python
from cvd_risk import CVDRiskAssessor

# Initialize assessor
assessor = CVDRiskAssessor()

# Prepare patient data
patient_data = {
    'age': 55,
    'sex': 1,
    'sbp': 140,
    'dbp': 90,
    'bmi': 28.5,
    # ... additional features
}

# Assess risk
risk_score = assessor.predict_risk(patient_data)
print(f"10-year CVD Risk: {risk_score:.2%}")
```

## 📈 Key Findings

### 1. Signal Processing Innovations
- Zero-phase Butterworth filtering preserves waveform morphology
- Comprehensive noise simulation improves real-world robustness
- 40% reduction in noise-related errors compared to unfiltered signals
- 95% accuracy in feature detection under severe noise conditions

### 2. SMOTE Effectiveness
- Improved CVD model accuracy from 65% to 78.66%
- Enhanced sensitivity for minority class (CVD cases)
- Maintained specificity while improving recall

### 3. Clinical Validation
- Blood pressure estimates meet ANSI/AAMI/ISO 81060-2:2019 standards
- Mean absolute errors within clinical acceptance thresholds
- Bland-Altman analysis confirms systematic agreement

### 4. Multi-Model Approach
- Ensemble methods consistently outperform single models
- Feature importance rankings provide clinical interpretability
- Cross-validation confirms robustness across patient populations

## 📁 Project Structure

```
AI-Enhanced-Blood-Pressure-Monitoring/
├── AI models/
│   ├── CVD risk predication model/
│   └── BP prediction model/
├── Individual_Design_Report.pdf
├── GDPgroup12_FinalReport_compressed.pdf
├── GDPgroup12_ProjectSpec.pdf
├── README.md
└── requirements.txt
```

## 📝 Research Methodology

### Development Process
1. **Literature Review**: Investigated cuffless BP monitoring techniques
2. **Dataset Selection**: Evaluated and obtained MIMIC-III and Framingham datasets
3. **Signal Processing**: Developed filtering and feature extraction pipeline
4. **Model Development**: Implemented and compared multiple ML architectures
5. **Validation**: Clinical-standard performance evaluation

### Key Technical Decisions
- **Piezoresistive sensors**: Chosen over PPG for reduced power consumption
- **Butterworth filter**: Selected for flat passband response
- **SMOTE**: Addressed critical class imbalance in medical data
- **Ensemble methods**: Provided superior performance and interpretability

## 🔮 Future Work

- **Multi-Signal Integration**: Combine ECG and PPG with pressure signals
- **Real-Time Optimization**: Reduce processing latency below 100ms
- **Memory Optimization**: Target 256MB usage for embedded deployment
- **Embedded Deployment**: Optimize for medical-grade wearable devices
- **Irregular Rhythm Handling**: Improve accuracy in arrhythmic cases (15-20% potential improvement)
- **Extended Validation**: Test with diverse patient demographics

## ⚠️ Limitations

- Development used MIMIC-III ABP signals (not piezoresistive sensor data)
- Limited testing on diverse patient populations
- Real-time performance not yet evaluated on embedded hardware
- Irregular cardiac rhythms may affect accuracy
- Requires further clinical validation for medical device approval

## 🎯 Applications

- **Wearable Health Monitoring**: Continuous BP tracking in daily life
- **Cardiovascular Risk Screening**: Early detection of CVD risk factors
- **Clinical Research**: Long-term BP trend analysis
- **Telemedicine**: Remote patient monitoring
- **Preventive Healthcare**: Proactive intervention strategies

## 👥 Team

**Group 12 - University of Southampton MEng Project (2024-2025)**

**Data Processing & AI Lead**: Mukilan Rajapandian
- CVD Risk Assessment Model Development
- Blood Pressure Estimation System
- Signal Processing Pipeline Implementation
- Machine Learning Model Optimization

**Academic Supervisors**: 
- Dr Rujie Sun (DHBE, University of Southampton)
- Professor Kai Yang (WSA, Customer)

**Full Team**:
- Oscar Robinson (Project Manager, Sensors)
- Mikayla Colegrave (Ethics, Sensors)
- Luqmanul Mohd Awallizam (Communications)
- Geethaartha Vagga (Mobile App Development)
- Mukilan Rajapandian (Data Analysis & ML/AI)

## 🙏 Acknowledgments

- University of Southampton for project supervision
- PhysioNet for MIMIC-III database access
- Framingham Heart Study and Kaggle for CVD dataset
- All team members for collaborative development

## 📚 References

1. Liu et al., "Cuffless Blood Pressure Estimation Using Pressure Pulse Wave Signals", Sensors, 2018
2. Huang et al., "A Highly Sensitive Pressure-Sensing Array for Blood Pressure Estimation", Sensors, 2019
3. MIMIC-III Waveform Database, PhysioNet, 2020
4. Framingham Heart Study Dataset, Kaggle
5. ANSI/AAMI/ISO 81060-2:2019 Standards

For complete references, see Individual_Design_Report.pdf and GDPgroup12_FinalReport_compressed.pdf.

## 💬 Citation

If you use this work in your research, please cite:

```bibtex
@misc{mukilan2025bpmonitoring,
  author = {Mukilan Rajapandian and Team},
  title = {AI-Enhanced Blood Pressure Monitoring System},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/mukilan2101/AI-Enhanced-Blood-Pressure-Monitoring}}
}
```

## 📄 License

This project is part of an academic group design project at the University of Southampton. For production deployment, additional safety measures, testing, regulatory compliance, and appropriate licensing are required.

---

**Note**: This system is designed for research and educational purposes. Clinical deployment requires extensive validation, regulatory approval, and compliance with medical device standards.
