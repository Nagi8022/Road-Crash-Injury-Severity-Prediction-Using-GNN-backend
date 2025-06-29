#!/usr/bin/env python3
"""
Test script for the Road Crash Analytics API
This script tests all major endpoints to ensure they're working correctly
"""

import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000/api"

def test_health_check():
    """Test the health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Demo mode: {data.get('demo_mode', 'Unknown')}")
            print(f"   Services: {data.get('services', {})}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🔍 Testing single prediction...")
    
    # Sample prediction data
    prediction_data = {
        "speed_limit": 30,
        "road_type": "Single carriageway",
        "light_conditions": "Daylight",
        "weather_conditions": "Fine no high winds",
        "road_surface_conditions": "Dry",
        "junction_detail": "Not at junction or within 20 metres",
        "junction_control": "Give way or uncontrolled",
        "pedestrian_crossing_human_control": "None within 50 metres",
        "pedestrian_crossing_physical_facilities": "No physical crossing facilities within 50 metres",
        "carriageway_hazards": "None",
        "urban_or_rural_area": "Urban",
        "did_police_officer_attend_scene_of_accident": "No",
        "trunk_road_flag": "Non-trunk",
        "lsoa_of_accident_location": "E01000001"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/single",
            json=prediction_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Single prediction successful:")
            print(f"   Severity: {data['severity']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Demo Mode: {data.get('demo_mode', False)}")
            return True
        else:
            print(f"❌ Single prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single prediction error: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n🔍 Testing batch prediction...")
    
    # Create a simple CSV file for testing
    csv_content = """speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
30,Single carriageway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000001
40,Dual carriageway,Darkness - lights lit,Raining without high winds,Wet or damp,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000002"""
    
    try:
        files = {'file': ('test_batch.csv', csv_content, 'text/csv')}
        response = requests.post(f"{BASE_URL}/predict/batch", files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful:")
            print(f"   Total records: {data['total_records']}")
            print(f"   Summary: {data['summary']}")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return False

def test_analytics():
    """Test analytics endpoints"""
    print("\n🔍 Testing analytics endpoints...")
    
    try:
        # Test analytics overview
        response = requests.get(f"{BASE_URL}/analytics/overview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Analytics overview successful:")
            print(f"   Total predictions: {data['total_predictions']}")
            print(f"   Average confidence: {data['average_confidence']:.3f}")
        else:
            print(f"❌ Analytics overview failed: {response.status_code}")
            return False
        
        # Test severity distribution
        response = requests.get(f"{BASE_URL}/visualizations/severity-distribution", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Severity distribution successful: {data['chart_type']}")
        else:
            print(f"❌ Severity distribution failed: {response.status_code}")
            return False
        
        # Test feature importance
        response = requests.get(f"{BASE_URL}/visualizations/feature-importance", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Feature importance successful: {data['chart_type']}")
        else:
            print(f"❌ Feature importance failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Analytics error: {str(e)}")
        return False

def test_intelligent_ml():
    """Test intelligent ML endpoints"""
    print("\n🔍 Testing Intelligent ML endpoints...")
    
    try:
        # Test XAI explanation
        prediction_data = {
            "speed_limit": 30,
            "road_type": "Single carriageway",
            "light_conditions": "Daylight",
            "weather_conditions": "Fine no high winds",
            "road_surface_conditions": "Dry",
            "junction_detail": "Not at junction or within 20 metres",
            "junction_control": "Give way or uncontrolled",
            "pedestrian_crossing_human_control": "None within 50 metres",
            "pedestrian_crossing_physical_facilities": "No physical crossing facilities within 50 metres",
            "carriageway_hazards": "None",
            "urban_or_rural_area": "Urban",
            "did_police_officer_attend_scene_of_accident": "No",
            "trunk_road_flag": "Non-trunk",
            "lsoa_of_accident_location": "E01000001"
        }
        
        response = requests.post(f"{BASE_URL}/xai/explain", json=prediction_data, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ XAI explanation successful:")
            print(f"   SHAP: {data['explanations']['shap_explanation']['method']}")
            print(f"   LIME: {data['explanations']['lime_explanation']['method']}")
        else:
            print(f"❌ XAI explanation failed: {response.status_code}")
            return False
        
        # Test intelligent ML overview
        response = requests.get(f"{BASE_URL}/intelligent-ml/overview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Intelligent ML overview successful:")
            print(f"   Services: {data['status']}")
            print(f"   Recent activity: {data['recent_activity']}")
        else:
            print(f"❌ Intelligent ML overview failed: {response.status_code}")
            return False
        
        # Test drift detection
        csv_content = """speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
30,Single carriageway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000001"""
        
        files = {'file': ('test_drift.csv', csv_content, 'text/csv')}
        response = requests.post(f"{BASE_URL}/drift/detect", files=files, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Drift detection successful:")
            print(f"   Records analyzed: {data['records_analyzed']}")
        else:
            print(f"❌ Drift detection failed: {response.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Intelligent ML error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🚗 Road Crash Analytics API Test Suite")
    print("=" * 50)
    print(f"Testing API at: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Wait a moment for the API to be ready
    print("⏳ Waiting for API to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Analytics", test_analytics),
        ("Intelligent ML", test_intelligent_ml)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 