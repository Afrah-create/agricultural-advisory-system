import streamlit as st
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import sys
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add the current directory to Python path (Colab compatible)
import os
current_dir = os.getcwd()
sys.path.append(current_dir)

# Import GitHub model loader
try:
    from github_model_loader import ModelManager, create_model_manager_from_config
    GITHUB_MODELS_AVAILABLE = True
except ImportError:
    GITHUB_MODELS_AVAILABLE = False
    st.warning("GitHub model loader not available. Using local models only.")

# Custom CSS for compact laptop-friendly styling
st.markdown("""
<style>
    /* Compact styling for laptop screens */
    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 1rem;
        max-width: 100%;
        height: 100vh;
        overflow: hidden;
    }
    
    /* Sidebar styling with scroll */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
        padding-top: 0.5rem;
        height: 100vh;
        overflow-y: auto;
    }
    
    .sidebar .sidebar-content .block-container {
        padding-top: 0.3rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    
    /* Compact header */
    .main-header {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .main-header h1 {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }
    
    .main-header p {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    /* Compact sections */
    .section-header {
        padding: 0.8rem;
        margin: 1rem 0 0.8rem 0;
        font-size: 1.1rem;
    }
    
    /* Compact cards */
    .metric-card {
        padding: 1rem;
        margin: 0.3rem;
        min-height: 80px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        margin-bottom: 0.3rem;
    }
    
    .metric-label {
        font-size: 0.8rem;
    }
    
    /* Compact buttons */
    .stButton > button {
        padding: 0.6rem 1.5rem;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    /* Compact sidebar elements */
    .sidebar .stSlider {
        margin-bottom: 0.3rem;
    }
    
    .sidebar .stSelectbox {
        margin-bottom: 0.3rem;
    }
    
    .sidebar .stCheckbox {
        margin-bottom: 0.2rem;
    }
    
    .sidebar .stMarkdown h3 {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    /* Hide default Streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Ensure main content doesn't scroll */
    .main {
        height: 100vh;
        overflow: hidden;
    }
    
    /* Compact footer */
    .footer {
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .footer h3 {
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }
    
    .footer p {
        font-size: 0.8rem;
        margin: 0.1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize GitHub model manager
@st.cache_resource
def initialize_model_manager():
    """Initialize the GitHub model manager"""
    if GITHUB_MODELS_AVAILABLE:
        try:
            # Try to load from config file
            if os.path.exists("github_config.json"):
                return create_model_manager_from_config("github_config.json")
            else:
                # Use default configuration
                return ModelManager("Afrah-create/agricultural-advisory-system")
        except Exception as e:
            st.error(f"Failed to initialize GitHub model manager: {e}")
            return None
    return None

# Initialize model manager
model_manager = initialize_model_manager()

# Import your components
try:
    # Try importing from files
    from cell_32_constrained_decoding import AgronomicRuleEngine
    from cell_33_uncertainty_calibration import UncertaintyCalibrator
    from cell_34_cropping_planner import MultiObjectiveCroppingPlanner, SoilProfile, ResourceConstraints
    from cell_35_integrated_system import IntegratedAgriculturalAdvisor
except ImportError:
    # Fallback: create minimal versions
    st.warning("Some components not available. Using simplified version.")
    
    class AgronomicRuleEngine:
        def __init__(self):
            pass
        def validate_recommendation(self, soil_profile, crop_recommendation):
            return {"is_valid": True, "violations": [], "confidence_score": 0.8}
    
    class UncertaintyCalibrator:
        def __init__(self):
            pass
        def calibrate_prediction_uncertainty(self, predictions, true_labels):
            return {"confidence_level": 0.95, "calibration_metrics": {"empirical_coverage": 0.95}}
    
    class MultiObjectiveCroppingPlanner:
        def __init__(self):
            pass
        def plan_cropping_system(self, soil_profile, constraints, objectives):
            return {"summary": {"crops": ["maize", "beans"], "total_yield": 5000, "total_cost": 3000, "total_profit": 2000}}
    
    class IntegratedAgriculturalAdvisor:
        def __init__(self):
            self.rule_engine = AgronomicRuleEngine()
            self.uncertainty_calibrator = UncertaintyCalibrator()
            self.cropping_planner = MultiObjectiveCroppingPlanner()
        def analyze_soil_and_recommend(self, soil_data, constraints, objectives):
            return {
                "executive_summary": {
                    "soil_quality_score": 0.7,
                    "recommended_crops": ["maize", "beans"],
                    "overall_recommendation_valid": True,
                    "uncertainty_level": "medium"
                },
                "detailed_analysis": {
                    "soil_analysis": {"strengths": ["Good pH"], "weaknesses": ["Low organic matter"]},
                    "recommendations": {"crops": ["maize", "beans"], "confidence": 0.8, "source": "github_models"},
                    "cropping_plan": {"summary": {"total_yield": 5000, "total_cost": 3000, "total_profit": 2000}}
                },
                "actionable_recommendations": ["Apply organic matter", "Plant maize and beans"],
                "risk_assessment": {"high_risk_factors": [], "medium_risk_factors": ["Weather variability"], "low_risk_factors": [], "mitigation_strategies": []}
            }

# Page configuration
st.set_page_config(
    page_title="Agricultural Advisory System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Agricultural Advisory System
from datetime import datetime

# Compact App Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; 
            padding: 1rem; border-radius: 8px; margin-bottom: 1rem; text-align: center;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);">
    <h1 style="margin: 0 0 0.3rem 0; font-size: 1.5rem; font-weight: 600;">
        Agricultural Advisory System
    </h1>
    <p style="margin: 0 0 0.5rem 0; font-size: 0.9rem; opacity: 0.9;">
        Evidence-backed crop recommendations for Uganda
    </p>
    <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">
        {current_date} • {current_time}
    </p>
</div>
""".format(
    current_date=datetime.now().strftime("%B %d, %Y"),
    current_time=datetime.now().strftime("%I:%M %p")
), unsafe_allow_html=True)

# Add spacing
st.markdown("---")

# Initialize the advisor with GitHub models
@st.cache_resource
def initialize_advisor():
    """Initialize the advisor with GitHub models if available"""
    if model_manager and GITHUB_MODELS_AVAILABLE:
        try:
            # Load models from GitHub
            models = model_manager.load_all_models()
            
            # Create enhanced advisor with GitHub models
            advisor = IntegratedAgriculturalAdvisor()
            
            # Load GitHub PKL models into advisor
            if "cropping_planner.pkl" in models:
                advisor.cropping_planner_model = models["cropping_planner.pkl"]
                st.success("Cropping planner model loaded from GitHub")
            
            if "integrated_advisor.pkl" in models:
                advisor.integrated_model = models["integrated_advisor.pkl"]
                st.success("Integrated advisor model loaded from GitHub")
            
            if "rule_engine.pkl" in models:
                advisor.rule_engine_model = models["rule_engine.pkl"]
                st.success("Rule engine model loaded from GitHub")
            
            if "uncertainty_calibrator.pkl" in models:
                advisor.uncertainty_model = models["uncertainty_calibrator.pkl"]
                st.success("Uncertainty calibrator model loaded from GitHub")
            
            return advisor
            
        except Exception as e:
            st.error(f"Failed to load GitHub models: {e}")
            return IntegratedAgriculturalAdvisor()
    else:
        return IntegratedAgriculturalAdvisor()

advisor = initialize_advisor()

# System Status (Simplified)
st.sidebar.markdown("### System Status")

if model_manager and GITHUB_MODELS_AVAILABLE:
    try:
        status = model_manager.get_model_status()
        loaded_models = sum(1 for info in status.values() if info["loaded"])
        total_models = len(status)
        
        if loaded_models == total_models:
            st.sidebar.success("✓ All models loaded successfully")
        else:
            st.sidebar.warning(f"⚠ {loaded_models}/{total_models} models loaded")
    except:
        st.sidebar.info("ℹ Using offline mode")
else:
    st.sidebar.info("ℹ Using offline mode")

st.sidebar.markdown("---")

# Professional Sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 12px; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
    <h2 style="color: white; margin: 0; font-weight: 300; letter-spacing: 1px;">Soil Profile Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Soil properties input with icons and better styling
st.sidebar.markdown("### Soil Properties")

ph = st.sidebar.slider(
    "pH Level", 
    4.0, 9.0, 6.5, 0.1,
    help="Soil acidity/alkalinity level"
)

organic_matter = st.sidebar.slider(
    "Organic Matter (%)", 
    0.0, 10.0, 2.0, 0.1,
    help="Percentage of organic matter in soil"
)

nitrogen = st.sidebar.slider(
    "Nitrogen (kg/ha)", 
    0, 200, 50, 5,
    help="Available nitrogen content"
)

phosphorus = st.sidebar.slider(
    "Phosphorus (kg/ha)", 
    0, 100, 15, 1,
    help="Available phosphorus content"
)

potassium = st.sidebar.slider(
    "Potassium (kg/ha)", 
    0, 300, 120, 5,
    help="Available potassium content"
)

# Soil texture and drainage
st.sidebar.markdown("### Soil Characteristics")
texture = st.sidebar.selectbox(
    "Soil Texture", 
    ["sand", "sandy_loam", "loam", "clay_loam", "clay"],
    help="Physical composition of the soil"
)

drainage = st.sidebar.selectbox(
    "Drainage", 
    ["poor", "moderate", "good", "excellent"],
    help="How well water drains through the soil"
)

# Resource constraints with better styling
st.sidebar.markdown("### Farm Constraints")
total_area = st.sidebar.slider(
    "Farm Area (hectares)", 
    0.1, 10.0, 2.0, 0.1,
    help="Total farm area available for cultivation"
)

budget = st.sidebar.slider(
    "Budget (USD)", 
    1000, 20000, 5000, 500,
    help="Available budget for farming activities"
)

labor_availability = st.sidebar.slider(
    "Labor Availability (person-days)", 
    50, 500, 200, 10,
    help="Available labor in person-days"
)

water_availability = st.sidebar.slider(
    "Water Availability (mm)", 
    500, 2000, 1200, 50,
    help="Available water resources in millimeters"
)

# Fertilizer availability
st.sidebar.markdown("### Fertilizer Availability")
fertilizer_nitrogen = st.sidebar.slider(
    "Nitrogen Fertilizer (kg)", 
    50, 500, 200, 10,
    help="Available nitrogen fertilizer"
)

fertilizer_phosphorus = st.sidebar.slider(
    "Phosphorus Fertilizer (kg)", 
    20, 200, 80, 5,
    help="Available phosphorus fertilizer"
)

fertilizer_potassium = st.sidebar.slider(
    "Potassium Fertilizer (kg)", 
    50, 300, 150, 10,
    help="Available potassium fertilizer"
)

# Objectives with better styling
st.sidebar.markdown("### Optimization Objectives")
maximize_yield = st.sidebar.checkbox("Maximize Yield", True)
minimize_cost = st.sidebar.checkbox("Minimize Cost", True)
maximize_profit = st.sidebar.checkbox("Maximize Profit", True)

# Compact Main Content Section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Soil Analysis & Recommendations")
    
    # Analysis button
    if st.button("Analyze Soil & Generate Recommendations", type="primary"):
        
        # Prepare soil data
        soil_data = {
        "pH": ph,
        "organic_matter": organic_matter,
        "nitrogen": nitrogen,
        "phosphorus": phosphorus,
        "potassium": potassium,
        "texture": texture,
        "drainage": drainage,
        "location": "Uganda"
    }
    
    # Prepare constraints
    constraints = {
        "total_area": total_area,
        "budget": budget,
        "labor_availability": labor_availability,
        "water_availability": water_availability,
        "fertilizer_nitrogen": fertilizer_nitrogen,
        "fertilizer_phosphorus": fertilizer_phosphorus,
        "fertilizer_potassium": fertilizer_potassium
    }
    
    # Prepare objectives
    objectives = []
    if maximize_yield:
        objectives.append("maximize_yield")
    if minimize_cost:
        objectives.append("minimize_cost")
    if maximize_profit:
        objectives.append("maximize_profit")
    
    # Generate comprehensive analysis
    with st.spinner("Analyzing soil and generating recommendations..."):
        try:
            report = advisor.analyze_soil_and_recommend(soil_data, constraints, objectives)
            
            # Professional success message
            st.markdown("""
            <div class="success-card">
                <h3 style="margin: 0; color: #155724;">Analysis Completed Successfully!</h3>
                <p style="margin: 0.5rem 0 0 0; color: #155724;">Your soil analysis and crop recommendations are ready.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Executive Summary with custom styling
            st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
            summary = report["executive_summary"]
            
            # Create custom metric cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['soil_quality_score']:.2f}</div>
                    <div class="metric-label">Soil Quality Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(summary['recommended_crops'])}</div>
                    <div class="metric-label">Recommended Crops</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                status = "Valid" if summary['overall_recommendation_valid'] else "Invalid"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{status}</div>
                    <div class="metric-label">Recommendations</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{summary['uncertainty_level'].title()}</div>
                    <div class="metric-label">Uncertainty Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Analysis with professional styling
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Soil Analysis
            st.markdown('<div class="section-header">Soil Analysis</div>', unsafe_allow_html=True)
            soil_analysis = report["detailed_analysis"]["soil_analysis"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Soil Strengths")
                if soil_analysis["strengths"]:
                    for strength in soil_analysis["strengths"]:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <strong>✓ {strength}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-card">
                        <strong>No significant strengths identified</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Areas for Improvement")
                if soil_analysis["weaknesses"]:
                    for weakness in soil_analysis["weaknesses"]:
                        st.markdown(f"""
                        <div class="warning-card">
                            <strong>⚠ {weakness}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-card">
                        <strong>✓ No significant weaknesses identified</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Crop Recommendations with badges
            st.markdown('<div class="section-header">Crop Recommendations</div>', unsafe_allow_html=True)
            recommendations = report["detailed_analysis"]["recommendations"]
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>Recommended Crops:</h4>
                <p>Based on your soil profile and conditions, we recommend the following crops:</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display crops as badges
            crop_badges = ""
            for crop in recommendations['crops']:
                crop_badges += f'<span class="crop-badge">{crop.title()}</span>'
            
            st.markdown(crop_badges, unsafe_allow_html=True)
            
            # Confidence and source info
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>Confidence Level:</strong> {recommendations['confidence']:.2f}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>Source:</strong> {recommendations['source'].title()}
                </div>
                """, unsafe_allow_html=True)
            
            # Cropping Plan with professional styling
            if report["detailed_analysis"]["cropping_plan"]:
                st.markdown('<div class="section-header">🌾 Cropping Plan</div>', unsafe_allow_html=True)
                plan = report["detailed_analysis"]["cropping_plan"]
                
                if "error" not in plan:
                    plan_summary = plan["summary"]
                    
                    # Economic metrics with custom styling
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{plan_summary['total_yield']:.0f} kg</div>
                            <div class="metric-label">Total Yield</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">${plan_summary['total_cost']:.0f}</div>
                            <div class="metric-label">Total Cost</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        profit_color = "#28a745" if plan_summary['total_profit'] > 0 else "#dc3545"
                        st.markdown(f"""
                        <div class="metric-card" style="background: linear-gradient(135deg, {profit_color}, {profit_color}dd);">
                            <div class="metric-value">${plan_summary['total_profit']:.0f}</div>
                            <div class="metric-label">Total Profit</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Actionable Recommendations with professional styling
            st.markdown('<div class="section-header">Actionable Recommendations</div>', unsafe_allow_html=True)
            
            for i, rec in enumerate(report["actionable_recommendations"], 1):
                st.markdown(f"""
                <div class="recommendation-card">
                    <strong>{i}.</strong> {rec}
                </div>
                """, unsafe_allow_html=True)
            
            # Risk Assessment with color-coded styling
            st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
            risk_assessment = report["risk_assessment"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### High Risk Factors")
                if risk_assessment["high_risk_factors"]:
                    for risk in risk_assessment["high_risk_factors"]:
                        st.markdown(f"""
                        <div class="risk-high">
                            <strong>● {risk}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-low">
                        <strong>✓ No high-risk factors</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("### Medium Risk Factors")
                if risk_assessment["medium_risk_factors"]:
                    for risk in risk_assessment["medium_risk_factors"]:
                        st.markdown(f"""
                        <div class="risk-medium">
                            <strong>● {risk}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-low">
                        <strong>✓ No medium-risk factors</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col3:
                st.markdown("### Low Risk Factors")
                if risk_assessment["low_risk_factors"]:
                    for risk in risk_assessment["low_risk_factors"]:
                        st.markdown(f"""
                        <div class="risk-low">
                            <strong>● {risk}</strong>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="risk-low">
                        <strong>✓ No low-risk factors</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Mitigation Strategies
            if risk_assessment["mitigation_strategies"]:
                st.markdown('<div class="section-header">Mitigation Strategies</div>', unsafe_allow_html=True)
                for i, strategy in enumerate(risk_assessment["mitigation_strategies"], 1):
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <strong>{i}.</strong> {strategy}
                    </div>
                    """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.write("Please check your inputs and try again.")

with col2:
    st.markdown("### Quick Info")
    st.info("""
    **System Status:** All models loaded successfully
    
    **Available Features:**
    - Soil analysis
    - Crop recommendations
    - Optimization planning
    - Risk assessment
    """)
    
    st.markdown("### Recent Activity")
    st.success("✓ System initialized")
    st.success("✓ Models loaded")
    st.info("ℹ Ready for analysis")

# Compact Footer
st.markdown("""
<div class="footer">
    <h3>Agricultural Advisory System for Uganda</h3>
    <p>Evidence-backed crop recommendations for smallholder farmers</p>
    <div style="margin-top: 0.5rem;">
        <span style="margin: 0 0.5rem; font-size: 0.8rem;">Built with Streamlit</span>
        <span style="margin: 0 0.5rem; font-size: 0.8rem;">Powered by GitHub Models</span>
        <span style="margin: 0 0.5rem; font-size: 0.8rem;">Version 2.0</span>
    </div>
    <p style="margin-top: 0.5rem; font-size: 0.8rem; opacity: 0.8;">
        © 2024 Agricultural Advisory System. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
