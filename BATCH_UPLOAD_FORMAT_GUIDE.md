# Batch Upload Format Guide - Road Crash Analytics

## 🚨 IMPORTANT: Correct File Format Required

The batch upload functionality requires **exact column names** and **specific data values**. Use the sample files provided in the application to ensure compatibility.

## 📋 Required CSV Format

### Column Headers (Must be exactly as shown):
```csv
speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
```

### Sample CSV Data:
```csv
30,Single carriageway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000001
40,Dual carriageway,Darkness - lights lit,Raining without high winds,Wet or damp,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000002
70,Motorway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Rural,No,Trunk,E01000003
```

## 📋 Required JSON Format

### Structure:
```json
[
  {
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
]
```

## 🎯 Valid Data Values

### speed_limit
- **Type**: Integer
- **Range**: 20-70
- **Examples**: 30, 40, 50, 60, 70

### road_type
- **Values**: 
  - "Single carriageway"
  - "Dual carriageway"
  - "Motorway"
  - "Slip road"

### light_conditions
- **Values**:
  - "Daylight"
  - "Darkness - lights lit"
  - "Darkness - lights unlit"
  - "Darkness - no lighting"

### weather_conditions
- **Values**:
  - "Fine no high winds"
  - "Raining no high winds"
  - "Snowing no high winds"
  - "Fine with high winds"
  - "Raining with high winds"
  - "Snowing with high winds"
  - "Fog or mist"
  - "Other"

### road_surface_conditions
- **Values**:
  - "Dry"
  - "Wet or damp"
  - "Snow"
  - "Frost or ice"
  - "Flood over 3cm deep"

### junction_detail
- **Values**:
  - "Not at junction or within 20 metres"
  - "Roundabout"
  - "Mini-roundabout"
  - "T or staggered junction"
  - "Slip road"
  - "Crossroads"
  - "More than 4 arms (not roundabout)"
  - "Private drive or entrance"
  - "Other junction"

### junction_control
- **Values**:
  - "Give way or uncontrolled"
  - "Auto traffic signal"
  - "Stop sign"
  - "Manual traffic signal"
  - "Authorised person"

### pedestrian_crossing_human_control
- **Values**:
  - "None within 50 metres"
  - "Control by school crossing patrol"
  - "Control by other authorised person"

### pedestrian_crossing_physical_facilities
- **Values**:
  - "No physical crossing facilities within 50 metres"
  - "Zebra crossing"
  - "Pelican, puffin, toucan or similar non-junction pedestrian light crossing"
  - "Pedestrian phase at traffic signal junction"
  - "Footbridge or subway"
  - "Central refuge"

### carriageway_hazards
- **Values**:
  - "None"
  - "Vehicle load on the road"
  - "Other object on the road"
  - "Previous accident"
  - "Dog on the road"
  - "Other animal on the road"
  - "Pedestrian in carriageway - not injured"
  - "Any animal in carriageway (except ridden horse)"

### urban_or_rural_area
- **Values**:
  - "Urban"
  - "Rural"

### did_police_officer_attend_scene_of_accident
- **Values**:
  - "Yes"
  - "No"

### trunk_road_flag
- **Values**:
  - "Trunk"
  - "Non-trunk"

### lsoa_of_accident_location
- **Type**: String
- **Format**: E01000001 (LSOA code)
- **Examples**: E01000001, E01000002, E01000003

## 🚀 How to Use

### Step 1: Download Sample Files
1. Go to the **Batch Prediction** tab
2. Click **"Download Sample CSV"** or **"Download Sample JSON"**
3. Save the file to your computer

### Step 2: Prepare Your Data
1. Use the sample file as a template
2. Replace the sample data with your actual crash data
3. Ensure all required columns are present
4. Use only the valid values listed above

### Step 3: Upload and Process
1. Click **"Choose File"** or drag and drop your file
2. Click **"Process File"**
3. Wait for processing to complete
4. View and download results

## ⚠️ Common Issues and Solutions

### Issue: "No valid records found"
**Solution**: Check that your CSV has the exact column headers shown above

### Issue: "Invalid data format"
**Solution**: Ensure all values match the valid options listed above

### Issue: "File too large"
**Solution**: Reduce file size to under 10MB

### Issue: "Processing failed"
**Solution**: 
1. Use the sample files as templates
2. Check for missing required columns
3. Verify data values are valid

## 📊 Expected Results

After successful processing, you'll receive:
- **Total Records**: Number of records processed
- **Success Rate**: Percentage of successful predictions
- **Severity Breakdown**: Count of Slight, Serious, and Fatal predictions
- **Sample Results**: First 10 predictions with confidence scores
- **Download Option**: Full results in JSON format

## 🎯 Tips for Success

1. **Always use sample files** as starting templates
2. **Copy column headers exactly** - no extra spaces or typos
3. **Use valid values only** - check the lists above
4. **Test with small files first** - 5-10 records
5. **Check file encoding** - use UTF-8
6. **Remove any empty rows** at the end of CSV files

## 📞 Support

If you continue to have issues:
1. Download and use the exact sample files provided
2. Check the browser console for detailed error messages
3. Ensure the backend is running on port 8000
4. Try both CSV and JSON formats to see which works better

The batch upload now works reliably with the correct format! 🎉 