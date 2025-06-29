#!/usr/bin/env python3
"""
Test script to verify CSV processing works correctly
"""

import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.data_processor import DataProcessor

async def test_csv_processing():
    """Test CSV processing with sample data"""
    
    # Sample CSV content
    csv_content = """road_type,road_class,speed_limit,area_type,junction_location,junction_control,junction_detail,hazards,road_surface_conditions,vehicle_type,light_conditions,weather_conditions
Single carriageway,Class A,30,Urban,Not at junction,Give way or uncontrolled,Not at junction,None,Dry,Car,Daylight,Fine no high winds
Dual carriageway,Class A,70,Rural,Not at junction,Give way or uncontrolled,Not at junction,None,Wet or damp,Car,Daylight,Raining no high winds"""
    
    print("Testing CSV processing...")
    print(f"CSV content:\n{csv_content}")
    
    # Create data processor
    processor = DataProcessor()
    
    try:
        # Process CSV
        processed_data = await processor.process_batch_file(
            csv_content.encode('utf-8'), 
            'test.csv'
        )
        
        print(f"\n✅ Processing successful!")
        print(f"Processed {len(processed_data)} records")
        
        for i, record in enumerate(processed_data):
            print(f"\nRecord {i+1}:")
            for key, value in record.items():
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_csv_processing()) 