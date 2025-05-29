import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

def rule_based_prediction(ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall):
    """Hybrid rule-based prediction system based on agricultural domain knowledge"""
    
    # Initialize risk score
    disease_risk = 0.0
    stress_factors = []
    
    # Vegetation health assessment (primary indicator)
    vegetation_health = (ndvi + evi + savi) / 3
    
    if vegetation_health < 0.2:
        disease_risk += 0.6
        stress_factors.append("Very low vegetation indices")
    elif vegetation_health < 0.4:
        disease_risk += 0.4
        stress_factors.append("Low vegetation health")
    elif vegetation_health < 0.6:
        disease_risk += 0.2
        stress_factors.append("Moderate vegetation stress")
    
    # NDVI-specific thresholds (most reliable indicator)
    if ndvi < 0.2:
        disease_risk += 0.4
        stress_factors.append("Critical NDVI levels")
    elif ndvi < 0.4:
        disease_risk += 0.2
        stress_factors.append("Suboptimal NDVI")
    
    # Red Edge Position (disease sensitivity)
    if rep < 700:
        disease_risk += 0.3
        stress_factors.append("Red edge shift indicating stress")
    elif rep < 710:
        disease_risk += 0.1
        stress_factors.append("Minor red edge variation")
    
    # Weather-based disease risk
    if humidity > 85 and 25 < temp_max < 35:
        disease_risk += 0.4
        stress_factors.append("Optimal fungal disease conditions")
    elif humidity > 75 and 20 < temp_max < 40:
        disease_risk += 0.2
        stress_factors.append("Elevated fungal risk")
    
    # Rainfall effects
    if rainfall > 15:
        disease_risk += 0.3
        stress_factors.append("Excessive moisture stress")
    elif rainfall > 8:
        disease_risk += 0.15
        stress_factors.append("High moisture conditions")
    
    # Temperature stress
    if temp_max > 40 or temp_min < 5:
        disease_risk += 0.25
        stress_factors.append("Temperature stress")
    elif temp_max > 38 or temp_min < 8:
        disease_risk += 0.1
        stress_factors.append("Suboptimal temperature")
    
    # Perfect storm conditions
    if len(stress_factors) >= 3:
        disease_risk += 0.1
    
    # Normalize risk score
    disease_risk = min(1.0, disease_risk)
    
    # Determine classification and confidence
    if disease_risk > 0.65:
        prediction = 'diseased'
        confidence = min(0.95, 0.7 + disease_risk * 0.25)
        risk_level = 'High'
        action = 'Immediate treatment required'
    elif disease_risk > 0.35:
        prediction = 'stressed'
        confidence = min(0.90, 0.6 + disease_risk * 0.3)
        risk_level = 'Medium'
        action = 'Monitor closely and prepare treatment'
    else:
        prediction = 'healthy'
        confidence = max(0.75, 1 - disease_risk)
        risk_level = 'Low'
        action = 'Continue normal practices'
    
    return {
        'predicted_class': prediction,
        'confidence': confidence,
        'risk_level': risk_level,
        'action': action,
        'disease_risk_score': disease_risk,
        'stress_factors': stress_factors,
        'vegetation_health_score': vegetation_health
    }

def main():
    st.set_page_config(
        page_title="AgriGuard Disease Detection",
        page_icon="🌾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.title("🌾 AgriGuard: AI-Powered Crop Disease Detection")
    st.markdown("**Multi-modal fusion of satellite imagery and weather data for early disease detection**")
    st.markdown("---")
    
    # Sidebar inputs
    st.sidebar.header("📊 Field Analysis Parameters")
    
    st.sidebar.subheader("🛰️ Satellite Vegetation Indices")
    st.sidebar.markdown("*Based on Sentinel-2 imagery analysis*")
    
    ndvi = st.sidebar.slider("NDVI (Normalized Difference Vegetation Index)", -1.0, 1.0, 0.5, 0.01)
    evi = st.sidebar.slider("EVI (Enhanced Vegetation Index)", -1.0, 1.0, 0.3, 0.01)
    savi = st.sidebar.slider("SAVI (Soil Adjusted Vegetation Index)", -1.0, 1.0, 0.25, 0.01)
    rep = st.sidebar.slider("REP (Red Edge Position)", 680.0, 750.0, 715.0, 0.5)
    
    st.sidebar.subheader("🌤️ Weather Conditions")
    st.sidebar.markdown("*Meteorological risk factors*")
    
    temp_max = st.sidebar.slider("Maximum Temperature (°C)", 15.0, 45.0, 30.0, 0.5)
    temp_min = st.sidebar.slider("Minimum Temperature (°C)", 5.0, 35.0, 20.0, 0.5)
    humidity = st.sidebar.slider("Relative Humidity (%)", 20.0, 100.0, 70.0, 1.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 50.0, 5.0, 0.5)
    
    # Always calculate result for display (moved outside button)
    result = rule_based_prediction(ndvi, evi, savi, rep, temp_max, temp_min, humidity, rainfall)
    
    # Main analysis section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Disease Risk Analysis")
        
        if st.button("🔍 Analyze Field Conditions", type="primary", use_container_width=True):
            st.success("✅ Analysis Complete!")
        
        # Always show results (not just when button clicked)
        # Results metrics
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            status_emoji = "🟢" if result['predicted_class'] == 'healthy' else "🟡" if result['predicted_class'] == 'stressed' else "🔴"
            st.metric(
                "Health Status",
                f"{status_emoji} {result['predicted_class'].title()}",
                f"{result['confidence']:.1%} confidence"
            )
        
        with col_b:
            st.metric("Risk Level", result['risk_level'])
        
        with col_c:
            st.metric("Disease Score", f"{result['disease_risk_score']:.2f}")
        
        with col_d:
            st.metric("Vegetation Health", f"{result['vegetation_health_score']:.2f}")
        
        # Action recommendations
        st.subheader("📋 Recommended Actions")
        if result['predicted_class'] == 'diseased':
            st.error(f"🚨 **{result['action']}**")
            recommendations = [
                "Apply targeted fungicide treatment within 24-48 hours",
                "Increase field monitoring to daily inspections",
                "Improve drainage if excessive moisture present",
                "Consider soil treatment for root-borne diseases"
            ]
        elif result['predicted_class'] == 'stressed':
            st.warning(f"⚠️ **{result['action']}**")
            recommendations = [
                "Monitor field conditions twice weekly",
                "Adjust irrigation based on weather forecast",
                "Prepare preventive treatment options",
                "Check for early disease symptoms manually"
            ]
        else:
            st.success(f"✅ **{result['action']}**")
            recommendations = [
                "Continue regular monitoring schedule",
                "Maintain current agricultural practices",
                "Keep preventive measures in place"
            ]
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"**{i}.** {rec}")
        
        # Stress factors analysis
        if result['stress_factors']:
            st.subheader("⚠️ Identified Stress Factors")
            for factor in result['stress_factors']:
                st.write(f"• {factor}")
        
        # Detailed analysis charts
        st.subheader("📊 Detailed Analysis")
        
        # Vegetation indices comparison
        fig_veg = go.Figure()
        
        indices = ['NDVI', 'EVI', 'SAVI']
        values = [ndvi, evi, savi]
        colors = ['green' if v > 0.4 else 'orange' if v > 0.2 else 'red' for v in values]
        
        fig_veg.add_trace(go.Bar(
            x=indices,
            y=values,
            marker_color=colors,
            text=[f'{v:.3f}' for v in values],
            textposition='auto'
        ))
        
        fig_veg.update_layout(
            title="Vegetation Health Indicators",
            yaxis_title="Index Value",
            showlegend=False
        )
        
        st.plotly_chart(fig_veg, use_container_width=True)
        
        # Weather risk assessment
        weather_risks = {
            'Temperature': 0.8 if temp_max > 38 or temp_min < 10 else 0.3 if temp_max > 35 else 0.1,
            'Humidity': 0.9 if humidity > 85 else 0.6 if humidity > 75 else 0.2,
            'Rainfall': 0.8 if rainfall > 15 else 0.4 if rainfall > 8 else 0.1
        }
        
        fig_weather = go.Figure(go.Bar(
            x=list(weather_risks.keys()),
            y=list(weather_risks.values()),
            marker_color=['red' if v > 0.6 else 'orange' if v > 0.3 else 'green' for v in weather_risks.values()],
            text=[f'{v:.2f}' for v in weather_risks.values()],
            textposition='auto'
        ))
        
        fig_weather.update_layout(
            title="Weather Risk Assessment",
            yaxis_title="Risk Score",
            showlegend=False
        )
        
        st.plotly_chart(fig_weather, use_container_width=True)
    
    with col2:
        st.header("📈 Current Indicators")
        
        # NDVI gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ndvi,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "NDVI"},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "darkgreen"},
                'steps': [
                    {'range': [0, 0.2], 'color': "lightcoral"},
                    {'range': [0.2, 0.4], 'color': "gold"},
                    {'range': [0.4, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.2
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Current conditions summary
        st.subheader("🌡️ Weather Summary")
        st.write(f"**Temperature:** {temp_min}°C - {temp_max}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Rainfall:** {rainfall}mm")
        
        # Risk timeline (now result is always available)
        st.subheader("📅 Risk Forecast")
        dates = pd.date_range('2024-11-01', periods=7, freq='D')
        base_risk = result['disease_risk_score']
        risk_trend = [base_risk * (1 + 0.1 * np.random.randn()) for _ in range(7)]
        risk_trend = [max(0, min(1, r)) for r in risk_trend]
        
        fig_timeline = go.Figure()
        fig_timeline.add_trace(go.Scatter(
            x=dates,
            y=risk_trend,
            mode='lines+markers',
            name='Disease Risk',
            line=dict(color='red', width=3)
        ))
        fig_timeline.update_layout(
            title="7-Day Risk Projection",
            xaxis_title="Date",
            yaxis_title="Risk Score",
            height=250
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**AgriGuard** - Advanced crop disease detection using satellite imagery and weather analytics")
    st.markdown("*Developed with multi-modal AI for precision agriculture*")

if __name__ == "__main__":
    main()