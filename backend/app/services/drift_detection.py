import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DriftDetector:
    """Data drift detection service"""
    
    def __init__(self):
        self.drift_thresholds = {
            'statistical_test_threshold': 0.05,
            'distribution_difference_threshold': 0.1,
            'feature_drift_threshold': 0.15,
            'overall_drift_threshold': 0.2
        }
        
        # Reference data (sample)
        self.reference_data = self._generate_sample_reference_data()
        self.drift_history = []
        
    def _generate_sample_reference_data(self) -> pd.DataFrame:
        """Generate sample reference data for drift detection"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], size=n_samples, p=[0.1, 0.3, 0.2, 0.2, 0.15, 0.05]),
            'road_type': np.random.choice(['Single carriageway', 'Dual carriageway', 'Motorway'], size=n_samples, p=[0.6, 0.3, 0.1]),
            'light_conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - no lighting'], size=n_samples, p=[0.7, 0.25, 0.05]),
            'weather_conditions': np.random.choice(['Fine no high winds', 'Raining without high winds', 'Snowing without high winds', 'Fog or mist'], size=n_samples, p=[0.6, 0.25, 0.1, 0.05]),
            'road_surface_conditions': np.random.choice(['Dry', 'Wet or damp', 'Snow', 'Ice'], size=n_samples, p=[0.7, 0.2, 0.08, 0.02]),
            'junction_detail': np.random.choice(['Not at junction or within 20 metres', 'Roundabout', 'T or staggered junction'], size=n_samples, p=[0.7, 0.2, 0.1]),
            'junction_control': np.random.choice(['Give way or uncontrolled', 'Auto traffic signal', 'Stop sign'], size=n_samples, p=[0.6, 0.3, 0.1]),
            'pedestrian_crossing_human_control': np.random.choice(['None within 50 metres', 'Control by school crossing patrol'], size=n_samples, p=[0.9, 0.1]),
            'pedestrian_crossing_physical_facilities': np.random.choice(['No physical crossing facilities within 50 metres', 'Zebra crossing', 'Pelican, puffin, toucan or similar non-junction light'], size=n_samples, p=[0.7, 0.2, 0.1]),
            'carriageway_hazards': np.random.choice(['None', 'Vehicle load on the road', 'Other object on the road'], size=n_samples, p=[0.8, 0.1, 0.1]),
            'urban_or_rural_area': np.random.choice(['Urban', 'Rural'], size=n_samples, p=[0.7, 0.3]),
            'did_police_officer_attend_scene_of_accident': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.1, 0.9]),
            'trunk_road_flag': np.random.choice(['Trunk', 'Non-trunk'], size=n_samples, p=[0.2, 0.8]),
            'lsoa_of_accident_location': [f'E010000{i:02d}' for i in np.random.randint(1, 100, size=n_samples)]
        }
        
        return pd.DataFrame(data)

    async def comprehensive_drift_analysis(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive drift analysis"""
        try:
            logger.info(f"Starting drift analysis for {len(current_data)} records")
            
            # Ensure current data has same columns as reference
            current_data = self._align_columns(current_data)
            
            # Perform different types of drift detection
            statistical_drift = await self._detect_statistical_drift(current_data)
            distribution_drift = await self._detect_distribution_drift(current_data)
            feature_drift = await self._detect_feature_drift(current_data)
            
            # Combine results
            alerts = []
            alert_count = 0
            
            # Add statistical drift alerts
            for feature, result in statistical_drift.items():
                if result['drift_detected']:
                    alerts.append({
                        'type': 'statistical_drift',
                        'feature': feature,
                        'severity': 'high' if result['p_value'] < 0.01 else 'medium',
                        'message': f"Statistical drift detected in {feature} (p-value: {result['p_value']:.4f})",
                        'details': result
                    })
                    alert_count += 1
            
            # Add distribution drift alerts
            for feature, result in distribution_drift.items():
                if result['drift_detected']:
                    alerts.append({
                        'type': 'distribution_drift',
                        'feature': feature,
                        'severity': 'high' if result['difference'] > 0.2 else 'medium',
                        'message': f"Distribution drift detected in {feature} (difference: {result['difference']:.3f})",
                        'details': result
                    })
                    alert_count += 1
            
            # Add feature drift alerts
            for feature, result in feature_drift.items():
                if result['drift_detected']:
                    alerts.append({
                        'type': 'feature_drift',
                        'feature': feature,
                        'severity': 'high' if result['drift_score'] > 0.3 else 'medium',
                        'message': f"Feature drift detected in {feature} (score: {result['drift_score']:.3f})",
                        'details': result
                    })
                    alert_count += 1
            
            # Calculate overall drift score
            overall_drift_score = self._calculate_overall_drift_score(statistical_drift, distribution_drift, feature_drift)
            
            # Determine recommendations
            recommendations = self._generate_recommendations(alerts, overall_drift_score)
            
            # Store in history
            drift_record = {
                'timestamp': datetime.now().isoformat(),
                'total_alerts': alert_count,
                'overall_drift_score': overall_drift_score,
                'alerts': alerts
            }
            self.drift_history.append(drift_record)
            
            # Limit history to last 100 records
            if len(self.drift_history) > 100:
                self.drift_history = self.drift_history[-100:]
            
            return {
                'alerts': alerts,
                'summary': {
                    'total_alerts': alert_count,
                    'critical_alerts': len([a for a in alerts if a['severity'] == 'high']),
                    'high_alerts': len([a for a in alerts if a['severity'] == 'high']),
                    'medium_alerts': len([a for a in alerts if a['severity'] == 'medium']),
                    'low_alerts': len([a for a in alerts if a['severity'] == 'low']),
                    'drift_types': list(set([a['type'] for a in alerts])),
                    'overall_drift_score': overall_drift_score,
                    'recommendations': recommendations
                },
                'detailed_results': {
                    'statistical_drift': statistical_drift,
                    'distribution_drift': distribution_drift,
                    'feature_drift': feature_drift
                }
            }
            
        except Exception as e:
            logger.error(f"Drift analysis failed: {str(e)}")
            return {
                'alerts': [],
                'summary': {
                    'total_alerts': 0,
                    'critical_alerts': 0,
                    'high_alerts': 0,
                    'medium_alerts': 0,
                    'low_alerts': 0,
                    'drift_types': [],
                    'overall_drift_score': 0.0,
                    'recommendations': ['Drift analysis failed - check data format']
                },
                'detailed_results': {}
            }

    def _align_columns(self, current_data: pd.DataFrame) -> pd.DataFrame:
        """Ensure current data has same columns as reference data"""
        reference_columns = self.reference_data.columns.tolist()
        
        # Add missing columns with default values
        for col in reference_columns:
            if col not in current_data.columns:
                if col == 'speed_limit':
                    current_data[col] = 30
                elif col in ['road_type', 'light_conditions', 'weather_conditions', 'road_surface_conditions']:
                    current_data[col] = current_data.iloc[0][col] if len(current_data) > 0 else 'Unknown'
                else:
                    current_data[col] = 'Unknown'
        
        # Select only reference columns
        return current_data[reference_columns]

    async def _detect_statistical_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical drift using hypothesis tests"""
        results = {}
        
        for column in self.reference_data.columns:
            try:
                ref_data = self.reference_data[column]
                curr_data = current_data[column]
                
                # Handle different data types
                if ref_data.dtype in ['int64', 'float64']:
                    # Numerical data - use t-test
                    if len(curr_data) > 1:
                        t_stat, p_value = stats.ttest_ind(ref_data, curr_data)
                        drift_detected = p_value < self.drift_thresholds['statistical_test_threshold']
                    else:
                        p_value = 1.0
                        drift_detected = False
                else:
                    # Categorical data - use chi-square test
                    ref_counts = ref_data.value_counts()
                    curr_counts = curr_data.value_counts()
                    
                    # Align categories
                    all_categories = set(ref_counts.index) | set(curr_counts.index)
                    ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
                    curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
                    
                    if len(all_categories) > 1 and curr_aligned.sum() > 0:
                        chi2_stat, p_value = stats.chi2_contingency([ref_aligned, curr_aligned])[:2]
                        drift_detected = p_value < self.drift_thresholds['statistical_test_threshold']
                    else:
                        p_value = 1.0
                        drift_detected = False
                
                results[column] = {
                    'drift_detected': drift_detected,
                    'p_value': p_value,
                    'test_type': 't_test' if ref_data.dtype in ['int64', 'float64'] else 'chi_square'
                }
                
            except Exception as e:
                logger.warning(f"Statistical drift detection failed for {column}: {str(e)}")
                results[column] = {
                    'drift_detected': False,
                    'p_value': 1.0,
                    'test_type': 'failed'
                }
        
        return results

    async def _detect_distribution_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect distribution drift using KL divergence"""
        results = {}
        
        for column in self.reference_data.columns:
            try:
                ref_data = self.reference_data[column]
                curr_data = current_data[column]
                
                if ref_data.dtype in ['int64', 'float64']:
                    # Numerical data - use histogram comparison
                    ref_hist, _ = np.histogram(ref_data, bins=10, density=True)
                    curr_hist, _ = np.histogram(curr_data, bins=10, density=True)
                    
                    # Calculate difference
                    difference = np.mean(np.abs(ref_hist - curr_hist))
                    drift_detected = difference > self.drift_thresholds['distribution_difference_threshold']
                    
                else:
                    # Categorical data - use proportion difference
                    ref_props = ref_data.value_counts(normalize=True)
                    curr_props = curr_data.value_counts(normalize=True)
                    
                    # Align categories
                    all_categories = set(ref_props.index) | set(curr_props.index)
                    ref_aligned = ref_props.reindex(all_categories, fill_value=0)
                    curr_aligned = curr_props.reindex(all_categories, fill_value=0)
                    
                    difference = np.mean(np.abs(ref_aligned - curr_aligned))
                    drift_detected = difference > self.drift_thresholds['distribution_difference_threshold']
                
                results[column] = {
                    'drift_detected': drift_detected,
                    'difference': float(difference),
                    'method': 'histogram' if ref_data.dtype in ['int64', 'float64'] else 'proportion'
                }
                
            except Exception as e:
                logger.warning(f"Distribution drift detection failed for {column}: {str(e)}")
                results[column] = {
                    'drift_detected': False,
                    'difference': 0.0,
                    'method': 'failed'
                }
        
        return results

    async def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect feature drift using domain adaptation methods"""
        results = {}
        
        try:
            # Combine reference and current data
            ref_data = self.reference_data.copy()
            curr_data = current_data.copy()
            
            ref_data['source'] = 'reference'
            curr_data['source'] = 'current'
            
            combined_data = pd.concat([ref_data, curr_data], ignore_index=True)
            
            # Encode categorical variables
            for column in combined_data.columns:
                if combined_data[column].dtype == 'object' and column != 'source':
                    combined_data[column] = combined_data[column].astype('category').cat.codes
            
            # Separate features and target
            X = combined_data.drop('source', axis=1)
            y = combined_data['source']
            
            # Train a simple classifier to detect drift
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score
            
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            scores = cross_val_score(clf, X, y, cv=3)
            
            # Calculate drift score (higher score = more drift)
            drift_score = np.mean(scores)
            
            # Check individual features
            for column in self.reference_data.columns:
                if column in X.columns:
                    # Train classifier on single feature
                    X_single = X[[column]]
                    scores_single = cross_val_score(clf, X_single, y, cv=3)
                    feature_drift_score = np.mean(scores_single)
                    
                    results[column] = {
                        'drift_detected': feature_drift_score > self.drift_thresholds['feature_drift_threshold'],
                        'drift_score': float(feature_drift_score),
                        'method': 'domain_classifier'
                    }
                else:
                    results[column] = {
                        'drift_detected': False,
                        'drift_score': 0.0,
                        'method': 'skipped'
                    }
            
        except Exception as e:
            logger.warning(f"Feature drift detection failed: {str(e)}")
            for column in self.reference_data.columns:
                results[column] = {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'method': 'failed'
                }
        
        return results

    def _calculate_overall_drift_score(self, statistical_drift: Dict, distribution_drift: Dict, feature_drift: Dict) -> float:
        """Calculate overall drift score"""
        try:
            scores = []
            
            # Statistical drift score
            stat_scores = [1 - result['p_value'] for result in statistical_drift.values() if result['p_value'] < 1.0]
            if stat_scores:
                scores.append(np.mean(stat_scores))
            
            # Distribution drift score
            dist_scores = [result['difference'] for result in distribution_drift.values()]
            if dist_scores:
                scores.append(np.mean(dist_scores))
            
            # Feature drift score
            feat_scores = [result['drift_score'] for result in feature_drift.values()]
            if feat_scores:
                scores.append(np.mean(feat_scores))
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception:
            return 0.0

    def _generate_recommendations(self, alerts: List[Dict], overall_drift_score: float) -> List[str]:
        """Generate recommendations based on drift analysis"""
        recommendations = []
        
        if overall_drift_score > 0.3:
            recommendations.append("High drift detected - consider retraining the model")
            recommendations.append("Review data collection processes")
        elif overall_drift_score > 0.2:
            recommendations.append("Moderate drift detected - monitor closely")
            recommendations.append("Consider incremental model updates")
        elif overall_drift_score > 0.1:
            recommendations.append("Low drift detected - continue monitoring")
        else:
            recommendations.append("No significant drift detected - continue normal operations")
        
        # Add specific recommendations based on alert types
        alert_types = [alert['type'] for alert in alerts]
        if 'statistical_drift' in alert_types:
            recommendations.append("Statistical drift detected - verify data quality")
        if 'distribution_drift' in alert_types:
            recommendations.append("Distribution drift detected - check for data pipeline changes")
        if 'feature_drift' in alert_types:
            recommendations.append("Feature drift detected - review feature engineering")
        
        return recommendations

    async def get_drift_history(self, days: int = 30) -> Dict[str, Any]:
        """Get drift detection history"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Filter recent history
            recent_history = [
                record for record in self.drift_history
                if datetime.fromisoformat(record['timestamp']) >= cutoff_date
            ]
            
            # Calculate statistics
            total_alerts = sum(record['total_alerts'] for record in recent_history)
            
            alerts_by_type = {}
            alerts_by_severity = {'high': 0, 'medium': 0, 'low': 0}
            
            for record in recent_history:
                for alert in record.get('alerts', []):
                    # Count by type
                    alert_type = alert['type']
                    alerts_by_type[alert_type] = alerts_by_type.get(alert_type, 0) + 1
                    
                    # Count by severity
                    severity = alert['severity']
                    alerts_by_severity[severity] = alerts_by_severity.get(severity, 0) + 1
            
            # Calculate trends
            if len(recent_history) > 1:
                drift_scores = [record['overall_drift_score'] for record in recent_history]
                change_rate = (drift_scores[-1] - drift_scores[0]) / len(drift_scores)
                
                if change_rate > 0.01:
                    trend = "increasing"
                elif change_rate < -0.01:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "insufficient_data"
                change_rate = 0.0
            
            return {
                'total_alerts': total_alerts,
                'alerts_by_type': alerts_by_type,
                'alerts_by_severity': alerts_by_severity,
                'trends': {
                    'trend': trend,
                    'change_rate': change_rate
                },
                'recent_records': recent_history[-10:]  # Last 10 records
            }
            
        except Exception as e:
            logger.error(f"Failed to get drift history: {str(e)}")
            return {
                'total_alerts': 0,
                'alerts_by_type': {},
                'alerts_by_severity': {},
                'trends': {'trend': 'unknown', 'change_rate': 0.0}
            }

    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Update drift detection thresholds"""
        self.drift_thresholds.update(new_thresholds)
        logger.info(f"Updated drift thresholds: {new_thresholds}") 