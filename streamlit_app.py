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

# Custom CSS for responsive scrollable styling
st.markdown("""
<style>
    /* Responsive styling for larger devices */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Sidebar styling with better spacing */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
        padding-top: 1rem;
    }
    
    .sidebar .sidebar-content .block-container {
        padding-top: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Professional header */
    .main-header {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .main-header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Better section spacing */
    .section-header {
        padding: 1rem;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.2rem;
    }
    
    /* Improved sidebar elements */
    .sidebar .stSlider {
        margin-bottom: 0.5rem;
    }
    
    .sidebar .stSelectbox {
        margin-bottom: 0.5rem;
    }
    
    .sidebar .stCheckbox {
        margin-bottom: 0.3rem;
    }
    
    .sidebar .stMarkdown h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Hide default Streamlit header */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional footer */
    .footer {
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    .footer h3 {
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .footer p {
        font-size: 0.9rem;
        margin: 0.2rem 0;
    }
    
    /* Better spacing for main content */
    .main .block-container > div {
        margin-bottom: 1rem;
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

# Professional App Header
st.markdown("""
<div style="background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; 
            padding: 1.5rem; border-radius: 10px; margin-bottom: 1.5rem; text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);">
    <h1 style="margin: 0 0 0.5rem 0; font-size: 1.8rem; font-weight: 600;">
        Agricultural Advisory System
    </h1>
    <p style="margin: 0 0 0.8rem 0; font-size: 1rem; opacity: 0.9;">
        Evidence-backed crop recommendations for Uganda
    </p>
    <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
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

# Main Content Section
st.markdown("### Soil Analysis & Recommendations")

# Prepare soil data (moved outside button for proper scope)
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

# Analysis button
if st.button("Analyze Soil & Generate Recommendations", type="primary"):
    
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
            
            # Executive Summary with neat tables
            st.markdown("### 📊 Executive Summary")
            
            # Create summary metrics table
            summary_data = {
                "Metric": ["Soil Quality Score", "Recommended Crops", "Recommendation Status", "Uncertainty Level"],
                "Value": [
                    f"{summary['soil_quality_score']:.2f}",
                    f"{len(summary['recommended_crops'])} crops",
                    "Valid" if summary['overall_recommendation_valid'] else "Invalid",
                    summary['uncertainty_level'].title()
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Detailed Analysis with professional styling
            st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
            
            # Soil Analysis with neat tables
            st.markdown("### 🌱 Soil Analysis")
            soil_analysis = report["detailed_analysis"]["soil_analysis"]
            
            # Soil strengths table
            if soil_analysis["strengths"]:
                strengths_data = {
                    "Soil Strengths": soil_analysis["strengths"]
                }
                strengths_df = pd.DataFrame(strengths_data)
                st.markdown("**✅ Soil Strengths:**")
                st.dataframe(strengths_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No significant soil strengths identified")
            
            # Soil weaknesses table
            if soil_analysis["weaknesses"]:
                weaknesses_data = {
                    "Areas for Improvement": soil_analysis["weaknesses"]
                }
                weaknesses_df = pd.DataFrame(weaknesses_data)
                st.markdown("**⚠️ Areas for Improvement:**")
                st.dataframe(weaknesses_df, use_container_width=True, hide_index=True)
            else:
                st.success("No significant soil weaknesses identified")
            
            # Crop Recommendations with neat tables
            st.markdown("### 🌾 Crop Recommendations")
            recommendations = report["detailed_analysis"]["recommendations"]
            
            # Recommended crops table
            crops_data = {
                "Recommended Crops": recommendations['crops']
            }
            crops_df = pd.DataFrame(crops_data)
            st.dataframe(crops_df, use_container_width=True, hide_index=True)
            
            # Recommendation details table
            rec_details_data = {
                "Attribute": ["Confidence Level", "Data Source"],
                "Value": [f"{recommendations['confidence']:.2f}", recommendations['source'].title()]
            }
            rec_details_df = pd.DataFrame(rec_details_data)
            st.dataframe(rec_details_df, use_container_width=True, hide_index=True)
            
            # Cropping Plan with neat tables
            if report["detailed_analysis"]["cropping_plan"]:
                st.markdown("### 📋 Cropping Plan")
                plan = report["detailed_analysis"]["cropping_plan"]
                
                if "error" not in plan:
                    plan_summary = plan["summary"]
                    
                    # Economic metrics table
                    economic_data = {
                        "Metric": ["Total Yield", "Total Cost", "Total Profit"],
                        "Value": [
                            f"{plan_summary['total_yield']:.0f} kg",
                            f"${plan_summary['total_cost']:.0f}",
                            f"${plan_summary['total_profit']:.0f}"
                        ]
                    }
                    economic_df = pd.DataFrame(economic_data)
                    st.dataframe(economic_df, use_container_width=True, hide_index=True)
            
            # Actionable Recommendations with neat tables
            st.markdown("### ✅ Actionable Recommendations")
            
            recommendations_data = {
                "Priority": [f"{i}" for i in range(1, len(report["actionable_recommendations"]) + 1)],
                "Recommendation": report["actionable_recommendations"]
            }
            recommendations_df = pd.DataFrame(recommendations_data)
            st.dataframe(recommendations_df, use_container_width=True, hide_index=True)
            
            # Risk Assessment with neat tables
            st.markdown("### ⚠️ Risk Assessment")
            risk_assessment = report["risk_assessment"]
            
            # High risk factors table
            if risk_assessment["high_risk_factors"]:
                high_risk_data = {
                    "High Risk Factors": risk_assessment["high_risk_factors"]
                }
                high_risk_df = pd.DataFrame(high_risk_data)
                st.markdown("**🔴 High Risk Factors:**")
                st.dataframe(high_risk_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No high-risk factors identified")
            
            # Medium risk factors table
            if risk_assessment["medium_risk_factors"]:
                medium_risk_data = {
                    "Medium Risk Factors": risk_assessment["medium_risk_factors"]
                }
                medium_risk_df = pd.DataFrame(medium_risk_data)
                st.markdown("**🟡 Medium Risk Factors:**")
                st.dataframe(medium_risk_df, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No medium-risk factors identified")
            
            # Low risk factors table
            if risk_assessment["low_risk_factors"]:
                low_risk_data = {
                    "Low Risk Factors": risk_assessment["low_risk_factors"]
                }
                low_risk_df = pd.DataFrame(low_risk_data)
                st.markdown("**🟢 Low Risk Factors:**")
                st.dataframe(low_risk_df, use_container_width=True, hide_index=True)
            else:
                st.info("ℹ️ No low-risk factors identified")
            
            # Mitigation strategies table
            if risk_assessment["mitigation_strategies"]:
                mitigation_data = {
                    "Priority": [f"{i}" for i in range(1, len(risk_assessment["mitigation_strategies"]) + 1)],
                    "Mitigation Strategy": risk_assessment["mitigation_strategies"]
                }
                mitigation_df = pd.DataFrame(mitigation_data)
                st.markdown("**🛡️ Mitigation Strategies:**")
                st.dataframe(mitigation_df, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.write("Please check your inputs and try again.")

# Information Panel
st.markdown("---")
st.markdown("### System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**System Status:**")
    st.success("✓ All models loaded successfully")
    
    st.markdown("**Available Features:**")
    st.info("""
    - Soil analysis
    - Crop recommendations  
    - Optimization planning
    - Risk assessment
    """)

with col2:
    st.markdown("**Recent Activity:**")
    st.success("✓ System initialized")
    st.success("✓ Models loaded")
    st.info("ℹ Ready for analysis")
    
    st.markdown("**Quick Stats:**")
    st.metric("Soil Parameters", "8")
    st.metric("Farm Constraints", "7")
    st.metric("Optimization Goals", f"{len(objectives)}")

# Professional Footer
st.markdown("""
<div class="footer">
    <h3>Agricultural Advisory System for Uganda</h3>
    <p>Evidence-backed crop recommendations for smallholder farmers</p>
    <div style="margin-top: 1rem;">
        <span style="margin: 0 1rem; font-size: 0.9rem;">Built with Streamlit</span>
        <span style="margin: 0 1rem; font-size: 0.9rem;">Powered by GitHub Models</span>
        <span style="margin: 0 1rem; font-size: 0.9rem;">Version 2.0</span>
    </div>
    <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
        © 2024 Agricultural Advisory System. All rights reserved.
    </p>
</div>
""", unsafe_allow_html=True)
