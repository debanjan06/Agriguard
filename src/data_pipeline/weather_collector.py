import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class WeatherDataCollector:
    def __init__(self, api_key=None):
        """Initialize weather data collector
        Get free API key from: https://openweathermap.org/api
        """
        self.api_key = api_key or "your_openweather_api_key"
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
    def get_historical_weather(self, lat, lon, start_date, end_date):
        """Get historical weather data for location"""
        
        # For now, simulate weather data (replace with real API calls)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        # Simulate realistic weather patterns for Karnataka
        np.random.seed(42)
        weather_data = []
        
        for date in dates:
            # Seasonal patterns
            month = date.month
            
            # Temperature (Celsius) - seasonal variation
            base_temp = 25 + 5 * np.sin(2 * np.pi * (month - 4) / 12)
            temp_max = base_temp + np.random.normal(5, 2)
            temp_min = base_temp - np.random.normal(5, 2)
            
            # Humidity (%) - monsoon patterns
            if month in [6, 7, 8, 9]:  # Monsoon months
                humidity = np.random.normal(80, 10)
            else:
                humidity = np.random.normal(60, 15)
            
            # Rainfall (mm)
            if month in [6, 7, 8, 9]:
                rainfall = np.random.exponential(5)
            else:
                rainfall = np.random.exponential(0.5)
            
            # Wind speed (km/h)
            wind_speed = np.random.normal(15, 5)
            
            weather_data.append({
                'date': date,
                'temp_max': max(temp_max, temp_min),
                'temp_min': min(temp_max, temp_min),
                'humidity': max(0, min(100, humidity)),
                'rainfall': max(0, rainfall),
                'wind_speed': max(0, wind_speed),
                'lat': lat,
                'lon': lon
            })
        
        return pd.DataFrame(weather_data)
    
    def calculate_disease_risk_factors(self, weather_df):
        """Calculate weather-based disease risk factors"""
        
        # Temperature stress
        weather_df['temp_stress'] = np.where(
            (weather_df['temp_max'] > 35) | (weather_df['temp_min'] < 10), 
            1, 0
        )
        
        # Humidity risk (high humidity promotes fungal diseases)
        weather_df['humidity_risk'] = np.where(
            weather_df['humidity'] > 75, 
            (weather_df['humidity'] - 75) / 25, 
            0
        )
        
        # Leaf wetness duration (rainfall + humidity)
        weather_df['leaf_wetness'] = (
            weather_df['rainfall'] * 0.3 + 
            weather_df['humidity'] * 0.01
        )
        
        # Disease favorability index
        weather_df['disease_favorability'] = (
            weather_df['humidity_risk'] * 0.4 +
            weather_df['leaf_wetness'] * 0.3 +
            weather_df['temp_stress'] * 0.3
        )
        
        return weather_df

# Test weather collector
if __name__ == "__main__":
    weather_collector = WeatherDataCollector()
    
    # Get weather for Mysore region
    mysore_weather = weather_collector.get_historical_weather(
        12.45, 76.75,  # Mysore coordinates
        '2024-06-01', '2024-12-01'
    )
    
    # Calculate risk factors
    mysore_weather = weather_collector.calculate_disease_risk_factors(mysore_weather)
    
    print("Weather data collected!")
    print(mysore_weather.head())
    print(f"Average disease favorability: {mysore_weather['disease_favorability'].mean():.3f}")