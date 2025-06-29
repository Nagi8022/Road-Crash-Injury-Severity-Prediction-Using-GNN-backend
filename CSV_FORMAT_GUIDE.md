# 🚗 CSV Upload Format Guide - Road Crash Analytics

## 📋 Required CSV Format

### Column Headers (Must be exactly as shown):
```csv
speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
```

## 📊 Sample CSV Data

```csv
speed_limit,road_type,light_conditions,weather_conditions,road_surface_conditions,junction_detail,junction_control,pedestrian_crossing_human_control,pedestrian_crossing_physical_facilities,carriageway_hazards,urban_or_rural_area,did_police_officer_attend_scene_of_accident,trunk_road_flag,lsoa_of_accident_location
30,Single carriageway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000001
40,Dual carriageway,Darkness - lights lit,Raining without high winds,Wet or damp,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000002
70,Motorway,Daylight,Fine no high winds,Dry,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Rural,No,Trunk,E01000003
50,Dual carriageway,Daylight,Fine no high winds,Dry,Roundabout,Auto traffic signal,None within 50 metres,Zebra crossing,None,Urban,Yes,Non-trunk,E01000004
30,Single carriageway,Darkness - lights lit,Snowing without high winds,Snow,Not at junction or within 20 metres,Give way or uncontrolled,None within 50 metres,No physical crossing facilities within 50 metres,None,Urban,No,Non-trunk,E01000005
60,Dual carriageway,Daylight,Raining without high winds,Wet or damp,T or staggered junction,Stop sign,None within 50 metres,No physical crossing facilities within 50 metres,Vehicle load on the road,Rural,No,Non-trunk,E01000006
```

## 📝 Field Descriptions and Valid Values

### 1. speed_limit
- **Type**: Integer
- **Valid Values**: 10, 20, 30, 40, 50, 60, 70, 80
- **Description**: Speed limit in mph

### 2. road_type
- **Type**: String
- **Valid Values**:
  - "Single carriageway"
  - "Dual carriageway"
  - "Motorway"
  - "Slip road"

### 3. light_conditions
- **Type**: String
- **Valid Values**:
  - "Daylight"
  - "Darkness - lights lit"
  - "Darkness - no lighting"
  - "Darkness - lights unlit"

### 4. weather_conditions
- **Type**: String
- **Valid Values**:
  - "Fine no high winds"
  - "Raining without high winds"
  - "Snowing without high winds"
  - "Fine with high winds"
  - "Raining with high winds"
  - "Snowing with high winds"
  - "Fog or mist"
  - "Other"

### 5. road_surface_conditions
- **Type**: String
- **Valid Values**:
  - "Dry"
  - "Wet or damp"
  - "Snow"
  - "Frost or ice"
  - "Flood over 3cm. deep"

### 6. junction_detail
- **Type**: String
- **Valid Values**:
  - "Not at junction or within 20 metres"
  - "Roundabout"
  - "Mini-roundabout"
  - "T or staggered junction"
  - "Slip road"
  - "Crossroads"
  - "More than 4 arms (not roundabout)"
  - "Private drive or entrance"
  - "Other junction"

### 7. junction_control
- **Type**: String
- **Valid Values**:
  - "Give way or uncontrolled"
  - "Auto traffic signal"
  - "Auto traffic signal and pelican"
  - "Stop sign"
  - "Give way to vehicle"
  - "Give way to pedestrian"
  - "Manual traffic signal"
  - "Pelican, puffin, toucan or similar non-junction light"
  - "Zebra crossing"
  - "School crossing patrol"
  - "Other"

### 8. pedestrian_crossing_human_control
- **Type**: String
- **Valid Values**:
  - "None within 50 metres"
  - "Control by school crossing patrol"
  - "Control by other authorised person"
  - "Control by police officer"
  - "Control by traffic warden"
  - "Other"

### 9. pedestrian_crossing_physical_facilities
- **Type**: String
- **Valid Values**:
  - "No physical crossing facilities within 50 metres"
  - "Zebra crossing"
  - "Pelican, puffin, toucan or similar non-junction light"
  - "Pedestrian phase at traffic signal junction"
  - "Footbridge or subway"
  - "Other"

### 10. carriageway_hazards
- **Type**: String
- **Valid Values**:
  - "None"
  - "Vehicle load on the road"
  - "Other object on the road"
  - "Previous accident"
  - "Dog on the road"
  - "Other animal on the road"
  - "Pedestrian in carriageway - not injured"
  - "Any animal in carriageway (except ridden horse)"

### 11. urban_or_rural_area
- **Type**: String
- **Valid Values**:
  - "Urban"
  - "Rural"

### 12. did_police_officer_attend_scene_of_accident
- **Type**: String
- **Valid Values**:
  - "Yes"
  - "No"

### 13. trunk_road_flag
- **Type**: String
- **Valid Values**:
  - "Trunk"
  - "Non-trunk"

### 14. lsoa_of_accident_location
- **Type**: String
- **Valid Values**: E01000001, E01000002, E01000003, E01000004, E01000005
- **Description**: LSOA (Lower Layer Super Output Area) code

## 🚀 How to Use

### Step 1: Create Your CSV File
1. Copy the header row exactly as shown above
2. Add your data rows using the valid values listed
3. Save as a `.csv` file with UTF-8 encoding

### Step 2: Upload to the System
1. Go to the **Batch Prediction** tab in the application
2. Click **"Choose File"** and select your CSV file
3. Click **"Process File"**
4. Wait for processing to complete

### Step 3: View Results
- **Total Records**: Number of records processed
- **Severity Breakdown**: Count of Slight, Serious, and Fatal predictions
- **Confidence Scores**: Individual prediction confidence levels
- **Download Results**: Get full results in JSON format

## ⚠️ Important Notes

### ✅ Do's:
- Use exact column headers (case-sensitive)
- Use only the valid values listed above
- Save file as UTF-8 encoding
- Include all 14 required columns
- Use commas as separators

### ❌ Don'ts:
- Don't add extra spaces in column names
- Don't use values not listed in the valid values
- Don't skip required columns
- Don't use different file encodings
- Don't add extra commas or quotes

## 🔧 Troubleshooting

### Common Issues:

1. **"No valid records found"**
   - Check column headers match exactly
   - Verify all values are from the valid lists above

2. **"Invalid data format"**
   - Ensure speed_limit is a number (10, 20, 30, etc.)
   - Check all string values match exactly

3. **"Missing required fields"**
   - Make sure all 14 columns are present
   - Check for typos in column names

4. **"File too large"**
   - Reduce file size to under 10MB
   - Split large files into smaller batches

## 📞 Support

If you continue to have issues:
1. Use the sample CSV provided in the application
2. Check the browser console for detailed error messages
3. Ensure the backend is running on port 8000
4. Try with a small test file first (5-10 records)

## 📊 Expected Results Format

After successful processing, you'll receive:
```json
{
  "message": "Batch prediction completed successfully",
  "total_records": 6,
  "results": [
    {
      "severity": "Slight",
      "confidence": 0.85,
      "risk_level": "Low",
      "model_agreement": true
    }
  ],
  "summary": {
    "slight": 4,
    "serious": 1,
    "fatal": 1
  }
}
```

---

**🎯 Pro Tip**: Always start with the sample CSV file provided in the application and modify it with your data to ensure compatibility! 