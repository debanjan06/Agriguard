import ee
import geemap
import os
from datetime import datetime, timedelta
import json

class AgriGuardDataCollector:
    def __init__(self):
        """Initialize Earth Engine and setup data collection"""
        ee.Initialize()
        self.s2_collection = 'COPERNICUS/S2_SR_HARMONIZED'
        self.landsat_collection = 'LANDSAT/LC08/C02/T1_L2'
        
    def define_study_areas(self):
        """Define agricultural regions in Karnataka for disease monitoring"""
        study_areas = {
            'mysore_tomato': ee.Geometry.Rectangle([76.5, 12.2, 77.0, 12.7]),
            'belgaum_cotton': ee.Geometry.Rectangle([74.4, 15.8, 75.0, 16.3]),
            'hassan_coffee': ee.Geometry.Rectangle([75.8, 13.0, 76.3, 13.5]),
            'mandya_sugarcane': ee.Geometry.Rectangle([76.8, 12.4, 77.3, 12.9])
        }
        return study_areas
    
    def calculate_vegetation_indices(self, image):
        """Calculate multiple vegetation indices for disease detection"""
        # NDVI - Normalized Difference Vegetation Index
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        
        # EVI - Enhanced Vegetation Index
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000),
                'BLUE': image.select('B2').divide(10000)
            }
        ).rename('EVI')
        
        # SAVI - Soil Adjusted Vegetation Index
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
            {
                'NIR': image.select('B8').divide(10000),
                'RED': image.select('B4').divide(10000)
            }
        ).rename('SAVI')
        
        # REP - Red Edge Position (disease sensitive)
        rep = image.expression(
            '700 + 40 * ((((RED + RE3) / 2) - RE1) / (RE2 - RE1))',
            {
                'RED': image.select('B4').divide(10000),
                'RE1': image.select('B5').divide(10000),
                'RE2': image.select('B6').divide(10000),
                'RE3': image.select('B7').divide(10000)
            }
        ).rename('REP')
        
        return image.addBands([ndvi, evi, savi, rep])
    
    def collect_temporal_data(self, geometry, start_date, end_date, crop_type='tomato'):
        """Collect time-series data for disease monitoring"""
        
        # Get Sentinel-2 collection
        collection = (ee.ImageCollection(self.s2_collection)
                     .filterBounds(geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 15))
                     .map(self.calculate_vegetation_indices))
        
        # Create composite for each month
        months = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current = start
        while current < end:
            month_start = current.strftime('%Y-%m-%d')
            month_end = (current.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
            month_end = month_end.strftime('%Y-%m-%d')
            
            monthly_composite = (collection
                               .filterDate(month_start, month_end)
                               .median()
                               .set('system:time_start', ee.Date(month_start).millis())
                               .set('month', current.month)
                               .set('year', current.year))
            
            months.append(monthly_composite)
            current = current.replace(day=1) + timedelta(days=32)
            current = current.replace(day=1)
        
        return ee.ImageCollection(months)
    
    def export_training_data(self, region_name, geometry, start_date, end_date):
        """Export data for model training"""
        
        # Collect temporal data
        time_series = self.collect_temporal_data(geometry, start_date, end_date)
        
        # Create export task
        task = ee.batch.Export.image.toDrive(
            image=time_series.toBands(),
            description=f'agriguard_{region_name}_{start_date}_{end_date}',
            folder='AgriGuard_Data',
            region=geometry,
            scale=10,  # 10m resolution
            crs='EPSG:4326',
            maxPixels=1e9
        )
        
        task.start()
        print(f"Export started for {region_name}")
        return task
    
    def simulate_disease_outbreak(self, healthy_timeseries, disease_type='blight'):
        """Simulate disease outbreak patterns for training data"""
        
        disease_params = {
            'blight': {'ndvi_drop': 0.3, 'evi_drop': 0.25, 'onset_speed': 14},
            'rust': {'ndvi_drop': 0.2, 'evi_drop': 0.15, 'onset_speed': 21},
            'mosaic': {'ndvi_drop': 0.4, 'evi_drop': 0.35, 'onset_speed': 7}
        }
        
        params = disease_params[disease_type]
        
        def apply_disease_effect(image):
            # Get image date
            date = ee.Date(image.get('system:time_start'))
            outbreak_date = ee.Date('2024-06-01')  # Simulate outbreak start
            
            # Calculate days since outbreak
            days_since = date.difference(outbreak_date, 'day')
            
            # Disease progression (sigmoid function)
            progression = ee.Image(1).divide(
                ee.Image(1).add(
                    ee.Image(-1).multiply(days_since.divide(params['onset_speed'])).exp()
                )
            )
            
            # Apply disease effects
            ndvi_diseased = image.select('NDVI').multiply(
                ee.Image(1).subtract(progression.multiply(params['ndvi_drop']))
            )
            
            evi_diseased = image.select('EVI').multiply(
                ee.Image(1).subtract(progression.multiply(params['evi_drop']))
            )
            
            return image.addBands([ndvi_diseased.rename('NDVI_diseased'), 
                                 evi_diseased.rename('EVI_diseased'),
                                 progression.rename('disease_severity')])
        
        return healthy_timeseries.map(apply_disease_effect)

# Test the collector
if __name__ == "__main__":
    collector = AgriGuardDataCollector()
    study_areas = collector.define_study_areas()
    
    # Test with Mysore tomato region
    mysore_region = study_areas['mysore_tomato']
    
    # Collect recent 6 months of data
    time_series = collector.collect_temporal_data(
        mysore_region, 
        '2024-06-01', 
        '2024-12-01', 
        'tomato'
    )
    
    print(f"Collected {time_series.size().getInfo()} monthly composites")
    
    # Start export (this will create files in Google Drive)
    # task = collector.export_training_data('mysore_tomato', mysore_region, '2024-06-01', '2024-12-01')
    
    print("Data collection setup complete!")
