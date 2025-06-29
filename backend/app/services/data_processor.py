import logging
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Union
from io import BytesIO, StringIO
import traceback
import os
from datetime import datetime

# Import the visualizer
from .data_visualizer import DataVisualizer

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing service for handling file uploads and data validation"""
    
    def __init__(self, reports_dir: str = 'reports'):
        """
        Initialize the DataProcessor
        
        Args:
            reports_dir: Directory to save generated reports
        """
        self.reports_dir = reports_dir
        os.makedirs(reports_dir, exist_ok=True)
        
        # Initialize visualizer
        self.visualizer = DataVisualizer(output_dir=reports_dir)
        
        # Define required columns
        self.required_columns = [
            'speed_limit', 'road_type', 'light_conditions', 'weather_conditions',
            'road_surface_conditions', 'junction_detail', 'junction_control',
            'pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities',
            'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident',
            'trunk_road_flag', 'lsoa_of_accident_location'
        ]
        
        self.column_types = {
            'speed_limit': 'int',
            'road_type': 'str',
            'light_conditions': 'str',
            'weather_conditions': 'str',
            'road_surface_conditions': 'str',
            'junction_detail': 'str',
            'junction_control': 'str',
            'pedestrian_crossing_human_control': 'str',
            'pedestrian_crossing_physical_facilities': 'str',
            'carriageway_hazards': 'str',
            'urban_or_rural_area': 'str',
            'did_police_officer_attend_scene_of_accident': 'str',
            'trunk_road_flag': 'str',
            'lsoa_of_accident_location': 'str'
        }
        
        self.valid_values = {
            'speed_limit': list(range(10, 81, 10)),  # 10 to 80 in steps of 10
            'road_type': ['Single carriageway', 'Dual carriageway', 'Motorway', 'Slip road'],
            'light_conditions': ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting', 'Darkness - lights unlit'],
            'weather_conditions': ['Fine no high winds', 'Raining without high winds', 'Snowing without high winds', 'Fine with high winds', 'Raining with high winds', 'Snowing with high winds', 'Fog or mist', 'Other'],
            'road_surface_conditions': ['Dry', 'Wet or damp', 'Snow', 'Frost or ice', 'Flood over 3cm. deep'],
            'junction_detail': ['Not at junction or within 20 metres', 'Roundabout', 'Mini-roundabout', 'T or staggered junction', 'Slip road', 'Crossroads', 'More than 4 arms (not roundabout)', 'Private drive or entrance', 'Other junction'],
            'junction_control': ['Give way or uncontrolled', 'Auto traffic signal', 'Auto traffic signal and pelican', 'Stop sign', 'Give way to vehicle', 'Give way to pedestrian', 'Manual traffic signal', 'Pelican, puffin, toucan or similar non-junction light', 'Zebra crossing', 'School crossing patrol', 'Other'],
            'pedestrian_crossing_human_control': ['None within 50 metres', 'Control by school crossing patrol', 'Control by other authorised person', 'Control by police officer', 'Control by traffic warden', 'Other'],
            'pedestrian_crossing_physical_facilities': ['No physical crossing facilities within 50 metres', 'Zebra crossing', 'Pelican, puffin, toucan or similar non-junction light', 'Pedestrian phase at traffic signal junction', 'Footbridge or subway', 'Other'],
            'carriageway_hazards': ['None', 'Vehicle load on the road', 'Other object on the road', 'Previous accident', 'Dog on the road', 'Other animal on the road', 'Pedestrian in carriageway - not injured', 'Any animal in carriageway (except ridden horse)'],
            'urban_or_rural_area': ['Urban', 'Rural'],
            'did_police_officer_attend_scene_of_accident': ['Yes', 'No'],
            'trunk_road_flag': ['Trunk', 'Non-trunk'],
            'lsoa_of_accident_location': ['E01000001', 'E01000002', 'E01000003', 'E01000004', 'E01000005']  # Sample LSOA codes
        }
        
    def create_sample_csv(self):
        """Generate sample CSV content for drift detection.
        
        Returns:
            str: CSV formatted string with sample data
        """
        import io
        import random
        from datetime import datetime, timedelta
        
        # Generate sample data
        num_samples = 100
        base_date = datetime.now() - timedelta(days=30)
        
        data = {
            'timestamp': [base_date + timedelta(hours=i) for i in range(num_samples)],
            'speed_limit': [random.choice(self.valid_values['speed_limit']) for _ in range(num_samples)],
            'road_type': [random.choice(self.valid_values['road_type']) for _ in range(num_samples)],
            'light_conditions': [random.choice(self.valid_values['light_conditions']) for _ in range(num_samples)],
            'weather_conditions': [random.choice(self.valid_values['weather_conditions']) for _ in range(num_samples)],
            'road_surface_conditions': [random.choice(self.valid_values['road_surface_conditions']) for _ in range(num_samples)],
            'junction_detail': [random.choice(self.valid_values['junction_detail']) for _ in range(num_samples)],
            'junction_control': [random.choice(self.valid_values['junction_control']) for _ in range(num_samples)],
            'pedestrian_crossing_human_control': [random.choice(self.valid_values['pedestrian_crossing_human_control']) for _ in range(num_samples)],
            'pedestrian_crossing_physical_facilities': [random.choice(self.valid_values['pedestrian_crossing_physical_facilities']) for _ in range(num_samples)],
            'carriageway_hazards': [random.choice(self.valid_values['carriageway_hazards']) for _ in range(num_samples)],
            'urban_or_rural_area': [random.choice(self.valid_values['urban_or_rural_area']) for _ in range(num_samples)],
            'did_police_officer_attend_scene_of_accident': [random.choice(self.valid_values['did_police_officer_attend_scene_of_accident']) for _ in range(num_samples)],
            'trunk_road_flag': [random.choice(self.valid_values['trunk_road_flag']) for _ in range(num_samples)],
            'lsoa_of_accident_location': [random.choice(self.valid_values['lsoa_of_accident_location']) for _ in range(num_samples)],
            'severity': [random.choice(['Slight', 'Serious', 'Fatal']) for _ in range(num_samples)],
            'is_weekend': [random.choice([0, 1]) for _ in range(num_samples)],
            'hour_of_day': [i % 24 for i in range(num_samples)],
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        
        return output.getvalue()

    async def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single input data with detailed error handling"""
        try:
            logger.debug("Validating input data")
            validated_data = {}
            
            # Process each field with validation
            for field in self.required_columns:
                value = input_data.get(field)
                
                # Handle missing or empty values
                if value is None or (isinstance(value, str) and not value.strip()):
                    if field == 'speed_limit':
                        validated_data[field] = 30
                        logger.debug(f"Missing {field}, using default 30")
                    else:
                        validated_data[field] = 'Unknown'
                        logger.debug(f"Missing {field}, using default 'Unknown'")
                    continue
                
                # Clean and validate the value
                try:
                    if field == 'speed_limit':
                        # Special handling for speed limit
                        try:
                            num_val = int(float(str(value).strip()))
                            # Ensure it's a multiple of 10 between 10 and 80
                            if num_val % 10 != 0 or num_val < 10 or num_val > 80:
                                logger.warning(f"Speed limit {num_val} not in valid range, using default 30")
                                validated_data[field] = 30
                            else:
                                validated_data[field] = num_val
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid speed_limit '{value}', using default 30")
                            validated_data[field] = 30
                    else:
                        # Handle string fields
                        str_val = str(value).strip()
                        
                        # Check against valid values if specified
                        if field in self.valid_values:
                            # Try exact match first
                            if str_val in self.valid_values[field]:
                                validated_data[field] = str_val
                                continue
                            
                            # Try case-insensitive match
                            str_val_lower = str_val.lower()
                            for valid_val in self.valid_values[field]:
                                if str(valid_val).lower() == str_val_lower:
                                    validated_data[field] = valid_val  # Use the canonical version
                                    logger.debug(f"Matched '{str_val}' to '{valid_val}' for {field}")
                                    break
                            else:
                                # No match found, use first valid value
                                logger.warning(f"Value '{str_val}' not in valid values for {field}, using default")
                                validated_data[field] = self.valid_values[field][0]
                        else:
                            validated_data[field] = str_val
                
                except Exception as e:
                    logger.warning(f"Error validating {field}='{value}': {str(e)}")
                    # Use defaults for validation errors
                    if field == 'speed_limit':
                        validated_data[field] = 30
                    else:
                        validated_data[field] = 'Unknown'
            
            logger.debug("Input validation successful")
            return validated_data
            
        except Exception as e:
            logger.error(f"Input validation failed: {str(e)}")
            raise

    async def process_batch_file(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """Process batch file (CSV or JSON)"""
        try:
            logger.info(f"Processing batch file: {filename}")
            logger.info(f"File size: {len(file_content)} bytes")
            
            # Determine file type
            if filename.lower().endswith('.csv'):
                return await self._process_csv_file(file_content)
            elif filename.lower().endswith('.json'):
                return await self._process_json_file(file_content)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
                
        except Exception as e:
            logger.error(f"Batch file processing failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _process_csv_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Process CSV file with robust error handling and format detection"""
        try:
            # Try different encodings
            encodings = ['utf-8-sig', 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            content_str = None
            
            for encoding in encodings:
                try:
                    content_str = file_content.decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content_str is None:
                raise ValueError("Could not decode file with any supported encoding. Please use UTF-8 or Latin-1 encoding.")
            
            # Clean up any problematic characters
            content_str = content_str.replace('\x00', '').strip()
            
            if not content_str:
                raise ValueError("The uploaded file is empty")
            
            # Try different CSV parsing options
            try:
                # First try with standard settings
                df = pd.read_csv(
                    StringIO(content_str),
                    dtype=str,  # Read all as strings initially
                    keep_default_na=False,  # Don't convert 'NA' to NaN
                    na_values=['', 'NA', 'N/A', 'NULL', 'NaN', 'nan'],
                    skip_blank_lines=True,
                    encoding_errors='replace'  # Replace invalid chars with placeholder
                )
            except Exception as e:
                logger.warning(f"Standard CSV parsing failed: {str(e)}, trying with different options")
                try:
                    # Try with different delimiter detection
                    df = pd.read_csv(
                        StringIO(content_str),
                        sep=None,  # Auto-detect delimiter
                        engine='python',
                        dtype=str,
                        keep_default_na=False,
                        encoding_errors='replace'
                    )
                except Exception as e2:
                    logger.error(f"CSV parsing failed with all options: {str(e2)}")
                    raise ValueError(f"Could not parse CSV file. Please ensure it's a valid CSV file. Error: {str(e2)}")
            
            # Clean column names: strip whitespace and normalize
            df.columns = [str(col).strip() for col in df.columns]
            
            # Remove completely empty rows and columns
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                raise ValueError("No data found in the CSV file after removing empty rows")
            
            logger.info(f"CSV loaded with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Validate and process
            return await self._process_dataframe(df)
            
        except Exception as e:
            logger.error(f"CSV processing failed: {str(e)}")
            raise

    async def _process_json_file(self, file_content: bytes) -> List[Dict[str, Any]]:
        """Process JSON file"""
        try:
            # Decode content with fallback encodings
            try:
                content_str = file_content.decode('utf-8')
            except UnicodeDecodeError:
                logger.warning("UTF-8 decoding failed, trying latin-1")
                content_str = file_content.decode('latin-1')
            
            # Parse JSON
            data = json.loads(content_str)
            logger.info(f"JSON loaded with {len(data)} records")
            
            # Convert to DataFrame if it's a list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])
            
            # Validate and process
            return await self._process_dataframe(df)
            
        except Exception as e:
            logger.error(f"JSON processing failed: {str(e)}")
            raise

    async def _process_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process DataFrame and validate records with detailed error handling"""
        try:
            logger.info(f"Processing DataFrame with {len(df)} rows")
            
            # Create a mapping of lowercase column names to original column names
            column_mapping = {str(col).lower(): str(col) for col in df.columns}
            processed_df = pd.DataFrame()
            missing_columns = []
            
            # Map and validate each required column
            for col in self.required_columns:
                col_lower = col.lower()
                
                # Try to find a matching column (case-insensitive)
                if col in df.columns:
                    # Exact match
                    matched_col = col
                elif col_lower in column_mapping:
                    # Case-insensitive match
                    matched_col = column_mapping[col_lower]
                    logger.warning(f"Matched column '{matched_col}' to expected column '{col}' (case-insensitive)")
                else:
                    # No match found
                    missing_columns.append(col)
                    continue
                
                # Add the column to the processed dataframe
                processed_df[col] = df[matched_col]
            
            # Add default values for missing columns
            if missing_columns:
                logger.warning(f"Missing required columns: {missing_columns}")
                for col in missing_columns:
                    if col == 'speed_limit':
                        processed_df[col] = 30
                        logger.warning(f"Added missing column '{col}' with default value 30")
                    else:
                        processed_df[col] = 'Unknown'
                        logger.warning(f"Added missing column '{col}' with default value 'Unknown'")
            
            # Ensure we only have the required columns in the correct order
            df = processed_df[self.required_columns]
            
            # Process and validate each row
            valid_records = []
            invalid_records = []
            
            logger.info(f"Processing DataFrame with columns: {df.columns.tolist()}")
            logger.info(f"First row sample: {df.iloc[0].to_dict() if len(df) > 0 else 'No data'}")
            
            for index, row in df.iterrows():
                try:
                    row_dict = row.to_dict()
                    logger.debug(f"Validating row {index + 1}: {row_dict}")
                    
                    # Clean and validate the row data
                    cleaned_row = {}
                    for col, value in row_dict.items():
                        # Handle missing/empty values
                        if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                            if col == 'speed_limit':
                                cleaned_row[col] = 30
                            else:
                                cleaned_row[col] = 'Unknown'
                            logger.debug(f"Row {index + 1}: Empty value for '{col}', using default")
                            continue
                        
                        # Clean the value
                        if isinstance(value, str):
                            value = value.strip()
                        
                        # Special handling for speed_limit
                        if col == 'speed_limit':
                            try:
                                # Convert to float first to handle decimal strings
                                num_val = float(str(value).strip())
                                # Round to nearest 10 and clamp to valid range
                                cleaned_row[col] = max(10, min(80, round(num_val / 10) * 10))
                                logger.debug(f"Row {index + 1}: Converted speed_limit '{value}' to {cleaned_row[col]}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Row {index + 1}: Invalid speed_limit '{value}', using default 30")
                                cleaned_row[col] = 30
                        else:
                            cleaned_row[col] = value
                    
                    # Validate the cleaned row
                    validated = await self.validate_input(cleaned_row)
                    valid_records.append(validated)
                    
                except Exception as e:
                    error_info = {
                        "row_number": index + 2,  # +2 for 1-based index and header row
                        "error": str(e),
                        "column_values": {k: str(v)[:100] for k, v in row_dict.items() if k in self.required_columns},
                        "message": f"Error in row {index + 2}: {str(e)}"
                    }
                    invalid_records.append(error_info)
                    logger.warning(f"Row {index + 2} validation failed: {error_info}")
                    logger.debug(f"Full error details: {traceback.format_exc()}")
                    continue
            
            logger.info(f"Processing complete: {len(valid_records)} valid records, {len(invalid_records)} invalid records")
            
            if not valid_records:
                # Prepare detailed error message with sample errors
                error_details = {
                    "total_records": len(df),
                    "valid_records": 0,
                    "invalid_records": len(invalid_records),
                    "sample_errors": invalid_records[:5],  # Include first 5 errors
                    "suggestion": "Please check your file format and data against the expected schema."
                }
                logger.error(f"All rows invalid. Error details: {json.dumps(error_details, indent=2)}")
                raise ValueError(
                    f"No valid records found in the uploaded file. "
                    f"Found {len(invalid_records)} invalid records. "
                    f"First error: {invalid_records[0]['error'] if invalid_records else 'Unknown error'}"
                )
            
            # Log summary of processing
            logger.info(f"Successfully processed {len(valid_records)} records")
            if invalid_records:
                logger.warning(f"Skipped {len(invalid_records)} invalid records")
            
            return valid_records
            
        except Exception as e:
            logger.error(f"DataFrame processing failed: {str(e)}")
            raise

    async def _validate_field(self, field: str, value: Any) -> Any:
        """Validate individual field"""
        try:
            # Type validation
            expected_type = self.column_types[field]
            
            if expected_type == 'int':
                # Convert to int accepting floats and numeric strings
                try:
                    if isinstance(value, (int, np.integer)):
                        validated_value = int(value)
                    elif isinstance(value, (float, np.floating)):
                        if np.isnan(value):
                            raise ValueError("nan value")
                        validated_value = int(round(value))
                    elif isinstance(value, str):
                        if value.strip() == "":
                            raise ValueError("blank value")
                        validated_value = int(float(value))  # handles "30" and "30.0"
                    else:
                        raise TypeError("unsupported type")
                except (ValueError, TypeError):
                    logger.warning(f"Field '{field}' could not be parsed as integer (value: {value}). Using default 30.")
                    validated_value = 30
                
                # Range validation for speed_limit
                if field == 'speed_limit':
                    if validated_value not in self.valid_values[field]:
                        logger.warning(f"Speed limit {validated_value} not in standard range, using default")
                        validated_value = 30
                
            elif expected_type == 'str':
                # Convert to string
                validated_value = str(value).strip()
                
                # Value validation for categorical fields
                if field in self.valid_values:
                    if validated_value not in self.valid_values[field]:
                        logger.warning(f"Value '{validated_value}' not in valid values for {field}, using default")
                        # Use first valid value as default
                        validated_value = self.valid_values[field][0]
            
            else:
                validated_value = value
            
            return validated_value
            
        except Exception as e:
            logger.error(f"Field validation failed for {field}: {str(e)}")
            raise

    async def process_batch_file(self, file_content: bytes, filename: str, generate_report: bool = False) -> Dict[str, Any]:
        """
        Process uploaded batch file (CSV/JSON) and optionally generate visual reports
        
        Args:
            file_content: Raw file content as bytes
            filename: Original filename (used to determine file type)
            generate_report: Whether to generate visualization reports
            
        Returns:
            Dict containing:
            - data: List of processed data records
            - report_path: Path to generated report (if generate_report=True)
            - stats: Summary statistics
            - plots: Dictionary of generated plots
        """
        result = {
            'data': [],
            'report_path': None,
            'stats': {},
            'plots': {}
        }
        try:
            if filename.lower().endswith('.csv'):
                # Read CSV file
                df = pd.read_csv(BytesIO(file_content))
            elif filename.lower().endswith(('.json', '.jsonl')):
                # Read JSON file
                try:
                    df = pd.read_json(BytesIO(file_content))
                except ValueError:
                    # Try reading as JSON Lines format
                    df = pd.read_json(BytesIO(file_content), lines=True)
            else:
                raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")
            
            # Convert column names to lowercase and strip whitespace
            df.columns = df.columns.str.lower().str.strip()
            
            # Check for missing required columns
            missing_columns = [col for col in self.required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
            
            # Convert data types and validate values
            processed_data = []
            for _, row in df.iterrows():
                processed_row = {}
                for col in self.required_columns:
                    value = row[col] if col in row and pd.notna(row[col]) else None
                    
                    # Convert to correct type
                    if col in self.column_types:
                        if self.column_types[col] == 'int' and value is not None:
                            try:
                                value = int(float(value))
                            except (ValueError, TypeError):
                                value = None
                    
                    # Validate against allowed values if specified
                    if col in self.valid_values and value is not None:
                        if value not in self.valid_values[col]:
                            value = None  # Or set to a default value
                    
                    processed_row[col] = value
                
                processed_data.append(processed_row)
            
            result['data'] = processed_data
            
            # Generate visualizations and reports if requested
            if generate_report and not df.empty:
                try:
                    # Generate summary statistics
                    result['stats'] = self.visualizer.generate_summary_statistics(df)
                    
                    # Generate plots
                    result['plots'] = {
                        'severity': self.visualizer.plot_severity_distribution(df),
                        'trend': self.visualizer.plot_trend_over_time(df)
                    }
                    
                    # Add feature distributions
                    for feature in ['speed_limit', 'road_type', 'light_conditions', 'weather_conditions']:
                        if feature in df.columns:
                            result['plots'][f'{feature}_dist'] = self.visualizer.plot_feature_distribution(df, feature)
                    
                    # Generate HTML report
                    report_filename = f"accident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                    report_path = os.path.join(self.reports_dir, report_filename)
                    
                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(self._generate_html_report(df, result['stats'], result['plots']))
                    
                    result['report_path'] = report_path
                    
                except Exception as e:
                    logger.error(f"Error generating report: {str(e)}")
                    logger.error(traceback.format_exc())
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing batch file: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def create_sample_csv(self) -> str:
        """Create a sample CSV file for users"""
        sample_data = [
            {
                'speed_limit': 30,
                'road_type': 'Single carriageway',
                'light_conditions': 'Daylight',
                'weather_conditions': 'Fine no high winds',
                'road_surface_conditions': 'Dry',
                'junction_detail': 'Not at junction or within 20 metres',
                'junction_control': 'Give way or uncontrolled',
                'pedestrian_crossing_human_control': 'None within 50 metres',
                'pedestrian_crossing_physical_facilities': 'No physical crossing facilities within 50 metres',
                'carriageway_hazards': 'None',
                'urban_or_rural_area': 'Urban',
                'did_police_officer_attend_scene_of_accident': 'No',
                'trunk_road_flag': 'Non-trunk',
                'lsoa_of_accident_location': 'E01000001'
            },
            {
                'speed_limit': 50,
                'road_type': 'Dual carriageway',
                'light_conditions': 'Darkness - lights lit',
                'weather_conditions': 'Raining without high winds',
                'road_surface_conditions': 'Wet or damp',
                'junction_detail': 'Roundabout',
                'junction_control': 'Auto traffic signal',
                'pedestrian_crossing_human_control': 'None within 50 metres',
                'pedestrian_crossing_physical_facilities': 'Zebra crossing',
                'carriageway_hazards': 'None',
                'urban_or_rural_area': 'Urban',
                'did_police_officer_attend_scene_of_accident': 'Yes',
                'trunk_road_flag': 'Non-trunk',
                'lsoa_of_accident_location': 'E01000002'
            },
            {
                'speed_limit': 70,
                'road_type': 'Motorway',
                'light_conditions': 'Daylight',
                'weather_conditions': 'Fine no high winds',
                'road_surface_conditions': 'Dry',
                'junction_detail': 'Not at junction or within 20 metres',
                'junction_control': 'Give way or uncontrolled',
                'pedestrian_crossing_human_control': 'None within 50 metres',
                'pedestrian_crossing_physical_facilities': 'No physical crossing facilities within 50 metres',
                'carriageway_hazards': 'None',
                'urban_or_rural_area': 'Rural',
                'did_police_officer_attend_scene_of_accident': 'No',
                'trunk_road_flag': 'Trunk',
                'lsoa_of_accident_location': 'E01000003'
            }
        ]
        
        df = pd.DataFrame(sample_data)
        return df.to_csv(index=False)

    def create_sample_json(self) -> str:
        """Create a sample JSON file for users"""
        sample_data = [
            {
                'speed_limit': 30,
                'road_type': 'Single carriageway',
                'light_conditions': 'Daylight',
                'weather_conditions': 'Fine no high winds',
                'road_surface_conditions': 'Dry',
                'junction_detail': 'Not at junction or within 20 metres',
                'junction_control': 'Give way or uncontrolled',
                'pedestrian_crossing_human_control': 'None within 50 metres',
                'pedestrian_crossing_physical_facilities': 'No physical crossing facilities within 50 metres',
                'carriageway_hazards': 'None',
                'urban_or_rural_area': 'Urban',
                'did_police_officer_attend_scene_of_accident': 'No',
                'trunk_road_flag': 'Non-trunk',
                'lsoa_of_accident_location': 'E01000001'
            },
            {
                'speed_limit': 50,
                'road_type': 'Dual carriageway',
                'light_conditions': 'Darkness - lights lit',
                'weather_conditions': 'Raining without high winds',
                'road_surface_conditions': 'Wet or damp',
                'junction_detail': 'Roundabout',
                'junction_control': 'Auto traffic signal',
                'pedestrian_crossing_human_control': 'None within 50 metres',
                'pedestrian_crossing_physical_facilities': 'Zebra crossing',
                'carriageway_hazards': 'None',
                'urban_or_rural_area': 'Urban',
                'did_police_officer_attend_scene_of_accident': 'Yes',
                'trunk_road_flag': 'Non-trunk',
                'lsoa_of_accident_location': 'E01000002'
            }
        ]
        
        return json.dumps(sample_data, indent=2)

    def _generate_html_report(self, df: pd.DataFrame, stats: Dict, plots: Dict) -> str:
        """Generate HTML report from data and visualizations"""
        # This is a simplified version - the full implementation is in DataVisualizer
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Accident Data Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .plot {{ margin: 20px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Accident Data Report</h1>
                <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Total records: {len(df):,}</p>
            </div>
        """
        
        # Add plots
        if 'severity' in plots and plots['severity']:
            html_content += f"""
            <div class="section">
                <h2>Accident Severity Distribution</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{plots['severity']}" 
                         alt="Severity Distribution" style="max-width: 80%;">
                </div>
            </div>
            """
            
        if 'trend' in plots and plots['trend']:
            html_content += f"""
            <div class="section">
                <h2>Accident Trend Over Time</h2>
                <div class="plot">
                    <img src="data:image/png;base64,{plots['trend']}" 
                         alt="Accident Trend" style="max-width: 80%;">
                </div>
            </div>
            """
            
        # Add statistics
        html_content += """
            <div class="section">
                <h2>Summary Statistics</h2>
                <pre>""" + json.dumps(stats, indent=2) + """</pre>
            </div>
        """
        
        # Close HTML
        html_content += """
            <div class="footer" style="margin-top: 50px; text-align: center; color: #666; font-size: 0.9em;">
                <p>Report generated by Road Safety Analytics System</p>
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def get_file_format_guide(self) -> Dict[str, Any]:
        """Get file format guide for users"""
        return {
            'supported_formats': ['CSV', 'JSON'],
            'required_columns': self.required_columns,
            'column_descriptions': {
                'speed_limit': 'Speed limit in mph (10, 20, 30, 40, 50, 60, 70)',
                'road_type': 'Type of road (Single carriageway, Dual carriageway, Motorway, Slip road)',
                'light_conditions': 'Lighting conditions (Daylight, Darkness - lights lit, Darkness - no lighting)',
                'weather_conditions': 'Weather conditions (Fine no high winds, Raining without high winds, etc.)',
                'road_surface_conditions': 'Road surface conditions (Dry, Wet or damp, Snow, Frost or ice)',
                'junction_detail': 'Junction details (Not at junction, Roundabout, T junction, etc.)',
                'junction_control': 'Junction control (Give way, Traffic signal, Stop sign, etc.)',
                'pedestrian_crossing_human_control': 'Pedestrian crossing control (None, School patrol, etc.)',
                'pedestrian_crossing_physical_facilities': 'Physical crossing facilities (None, Zebra, Pelican, etc.)',
                'carriageway_hazards': 'Hazards on carriageway (None, Vehicle load, Object on road, etc.)',
                'urban_or_rural_area': 'Area type (Urban, Rural)',
                'did_police_officer_attend_scene_of_accident': 'Police attendance (Yes, No)',
                'trunk_road_flag': 'Trunk road flag (Trunk, Non-trunk)',
                'lsoa_of_accident_location': 'LSOA code (e.g., E01000001)'
            },
            'valid_values': self.valid_values,
            'example_csv': self.create_sample_csv(),
            'example_json': self.create_sample_json()
        }