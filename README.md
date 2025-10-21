# Agricultural Advisory System

A comprehensive agricultural advisory system that provides evidence-backed crop recommendations and cropping plans for smallholder farmers in Uganda.

## 🌾 Features

- **Soil Analysis**: Comprehensive soil profile analysis with pH, nutrients, and texture assessment
- **Crop Recommendations**: AI-powered crop recommendations based on soil conditions
- **Cropping Plans**: Multi-objective optimization for yield, cost, and profit
- **Risk Assessment**: Identification and mitigation of agricultural risks
- **GitHub Model Integration**: Load trained models directly from GitHub repositories

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run streamlit_app.py
```

### 3. Configure GitHub Models

1. Enter your GitHub repository in the sidebar
2. Click "🔄 Refresh Models from GitHub"
3. Start using the agricultural advisory system!

## 📁 Repository Structure

```
agricultural-advisory-system/
├── streamlit_app.py              # Main Streamlit application
├── github_model_loader.py        # GitHub model loading utility
├── github_config.json           # Configuration file
├── requirements.txt             # Python dependencies
├── models/                      # Local model storage (optional)
│   ├── cropping_planner.pkl     # Cropping planner model
│   ├── integrated_advisor.pkl   # Integrated advisor model
│   ├── rule_engine.pkl         # Rule engine model
│   └── uncertainty_calibrator.pkl # Uncertainty calibrator model
└── README.md                    # This file
```

## 🔧 Configuration

The system uses `github_config.json` to configure GitHub model loading:

```json
{
  "github_repo": "Afrah-create/agricultural-advisory-system",
  "branch": "main",
  "token": null,
  "models": {
    "cropping_planner": "cropping_planner.pkl",
    "integrated_advisor": "integrated_advisor.pkl",
    "rule_engine": "rule_engine.pkl",
    "uncertainty_calibrator": "uncertainty_calibrator.pkl"
  }
}
```

## 🌱 How to Use

1. **Configure Soil Properties**: Set pH, organic matter, nutrients, texture, and drainage
2. **Set Farm Constraints**: Define area, budget, labor, and water availability
3. **Choose Objectives**: Select optimization goals (yield, cost, profit)
4. **Generate Analysis**: Click "Analyze Soil & Generate Recommendations"
5. **Review Results**: View comprehensive analysis and actionable recommendations

## 📊 Model Integration

The system automatically loads PKL models from your GitHub repository:

- **Cropping Planner**: Multi-objective optimization for crop planning
- **Integrated Advisor**: Comprehensive agricultural advisory system
- **Rule Engine**: Agronomic rules and constraints validation
- **Uncertainty Calibrator**: Prediction uncertainty quantification

## 🔄 Model Updates

To update models:

1. Push new PKL files to your GitHub repository
2. Click "🔄 Refresh Models from GitHub" in the app
3. Updated models will be loaded automatically

## 🛠️ Development

### Local Development

```bash
# Clone the repository
git clone https://github.com/Afrah-create/agricultural-advisory-system.git
cd agricultural-advisory-system

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### Model Training

Models are trained using the agricultural data processing pipeline and saved as PKL files for deployment.

## 📈 Performance

- **Model Caching**: Automatic caching for faster loading
- **Offline Support**: Fallback to local models when GitHub is unavailable
- **Error Handling**: Graceful degradation with informative error messages

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration examples

---

**Built with ❤️ for agricultural communities in Uganda** 🌾
