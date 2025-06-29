import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class XAIService:
    """Explainable AI service using SHAP and LIME"""
    
    def __init__(self, predictor):
        self.predictor = predictor
        self.shap_explainer = None
        self.lime_explainer = None
        self.feature_names = [
            'speed_limit', 'road_type', 'light_conditions', 'weather_conditions',
            'road_surface_conditions', 'junction_detail', 'junction_control',
            'pedestrian_crossing_human_control', 'pedestrian_crossing_physical_facilities',
            'carriageway_hazards', 'urban_or_rural_area', 'did_police_officer_attend_scene_of_accident',
            'trunk_road_flag', 'lsoa_of_accident_location'
        ]
        
        # Sample data for demo explanations
        self.sample_data = pd.DataFrame({
            'speed_limit': [30, 40, 50, 60, 70],
            'road_type': ['Single carriageway', 'Dual carriageway', 'Single carriageway', 'Motorway', 'Dual carriageway'],
            'light_conditions': ['Daylight', 'Darkness - lights lit', 'Daylight', 'Daylight', 'Darkness - lights lit'],
            'weather_conditions': ['Fine no high winds', 'Raining without high winds', 'Fine no high winds', 'Fine no high winds', 'Raining without high winds'],
            'road_surface_conditions': ['Dry', 'Wet or damp', 'Dry', 'Dry', 'Wet or damp'],
            'junction_detail': ['Not at junction or within 20 metres'] * 5,
            'junction_control': ['Give way or uncontrolled'] * 5,
            'pedestrian_crossing_human_control': ['None within 50 metres'] * 5,
            'pedestrian_crossing_physical_facilities': ['No physical crossing facilities within 50 metres'] * 5,
            'carriageway_hazards': ['None'] * 5,
            'urban_or_rural_area': ['Urban'] * 5,
            'did_police_officer_attend_scene_of_accident': ['No'] * 5,
            'trunk_road_flag': ['Non-trunk'] * 5,
            'lsoa_of_accident_location': ['E01000001', 'E01000002', 'E01000003', 'E01000004', 'E01000005']
        })

    async def initialize_explainers(self):
        """Initialize SHAP and LIME explainers"""
        try:
            # Try to initialize SHAP
            try:
                import shap
                # Create a simple tree explainer for demo
                from sklearn.ensemble import RandomForestClassifier
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                rf.fit(self.sample_data, np.random.choice([0, 1, 2], size=len(self.sample_data)))
                self.shap_explainer = shap.TreeExplainer(rf)
                logger.info("✅ SHAP explainer initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ SHAP initialization failed: {str(e)}")
                self.shap_explainer = None

            # Try to initialize LIME
            try:
                from lime.lime_tabular import LimeTabularExplainer
                self.lime_explainer = LimeTabularExplainer(
                    self.sample_data.values,
                    feature_names=self.feature_names,
                    class_names=['Slight', 'Serious', 'Fatal'],
                    mode='classification'
                )
                logger.info("✅ LIME explainer initialized successfully")
            except Exception as e:
                logger.warning(f"⚠️ LIME initialization failed: {str(e)}")
                self.lime_explainer = None

        except Exception as e:
            logger.error(f"❌ XAI initialization failed: {str(e)}")
            self.shap_explainer = None
            self.lime_explainer = None

    async def explain_prediction_shap(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate SHAP explanation for a prediction"""
        try:
            if self.shap_explainer is None:
                return self._generate_demo_shap_explanation(input_data)
            
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Generate SHAP values
            shap_values = self.shap_explainer.shap_values(df)
            
            # Get feature importance
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                if i < len(shap_values[0]):
                    feature_importance[feature] = float(shap_values[0][i])
            
            # Sort by absolute importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Generate explanation text
            top_features = sorted_features[:5]
            explanation_parts = []
            for feature, value in top_features:
                if abs(value) > 0.01:  # Only include significant features
                    if value > 0:
                        explanation_parts.append(f"{feature} increases risk")
                    else:
                        explanation_parts.append(f"{feature} decreases risk")
            
            explanation = ". ".join(explanation_parts) + "."
            
            return {
                "method": "SHAP",
                "feature_importance": dict(sorted_features),
                "explanation": explanation,
                "shap_values": shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
                "base_value": float(self.shap_explainer.expected_value) if hasattr(self.shap_explainer, 'expected_value') else 0.0
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            return self._generate_demo_shap_explanation(input_data)

    async def explain_prediction_lime(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate LIME explanation for a prediction"""
        try:
            if self.lime_explainer is None:
                return self._generate_demo_lime_explanation(input_data)
            
            # Convert input to numpy array
            input_array = np.array([list(input_data.values())])
            
            # Generate LIME explanation
            explanation = self.lime_explainer.explain_instance(
                input_array[0], 
                self._dummy_predict_proba,
                num_features=len(self.feature_names),
                top_labels=3
            )
            
            # Extract feature importance
            feature_importance = {}
            for feature, weight in explanation.as_list():
                feature_importance[feature] = float(weight)
            
            # Generate explanation text
            explanation_parts = []
            for feature, weight in explanation.as_list()[:5]:
                if abs(weight) > 0.01:
                    if weight > 0:
                        explanation_parts.append(f"{feature} contributes positively")
                    else:
                        explanation_parts.append(f"{feature} contributes negatively")
            
            explanation_text = ". ".join(explanation_parts) + "."
            
            return {
                "method": "LIME",
                "feature_importance": feature_importance,
                "explanation": explanation_text,
                "local_prediction": explanation.local_pred.tolist(),
                "score": explanation.score
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            return self._generate_demo_lime_explanation(input_data)

    async def get_explanation_comparison(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get both SHAP and LIME explanations for comparison"""
        try:
            shap_explanation = await self.explain_prediction_shap(input_data)
            lime_explanation = await self.explain_prediction_lime(input_data)
            
            return {
                "shap_explanation": shap_explanation,
                "lime_explanation": lime_explanation,
                "comparison": {
                    "methods_used": ["SHAP", "LIME"],
                    "agreement_level": self._calculate_agreement(shap_explanation, lime_explanation),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Explanation comparison failed: {str(e)}")
            return {
                "shap_explanation": self._generate_demo_shap_explanation(input_data),
                "lime_explanation": self._generate_demo_lime_explanation(input_data),
                "comparison": {
                    "methods_used": ["SHAP (Demo)", "LIME (Demo)"],
                    "agreement_level": "demo_mode",
                    "timestamp": datetime.now().isoformat()
                }
            }

    def _dummy_predict_proba(self, x):
        """Dummy prediction function for LIME"""
        # Return random probabilities for demo
        return np.array([[0.3, 0.5, 0.2]])

    def _generate_demo_shap_explanation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo SHAP explanation"""
        # Create realistic demo feature importance based on input
        feature_importance = {}
        
        # Speed limit effect
        speed_limit = input_data.get('speed_limit', 30)
        feature_importance['speed_limit'] = (speed_limit - 30) / 100
        
        # Weather effect
        weather = input_data.get('weather_conditions', 'Fine no high winds')
        if 'rain' in weather.lower() or 'snow' in weather.lower():
            feature_importance['weather_conditions'] = 0.15
        elif 'fog' in weather.lower():
            feature_importance['weather_conditions'] = 0.25
        else:
            feature_importance['weather_conditions'] = -0.05
        
        # Light conditions effect
        light = input_data.get('light_conditions', 'Daylight')
        if 'dark' in light.lower():
            feature_importance['light_conditions'] = 0.12
        else:
            feature_importance['light_conditions'] = -0.08
        
        # Road type effect
        road_type = input_data.get('road_type', 'Single carriageway')
        if 'motorway' in road_type.lower():
            feature_importance['road_type'] = 0.08
        elif 'dual' in road_type.lower():
            feature_importance['road_type'] = 0.05
        else:
            feature_importance['road_type'] = -0.03
        
        # Add some random variation for other features
        for feature in self.feature_names:
            if feature not in feature_importance:
                feature_importance[feature] = np.random.uniform(-0.1, 0.1)
        
        # Sort by absolute importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation text
        explanation_parts = []
        for feature, value in sorted_features[:3]:
            if abs(value) > 0.05:
                if value > 0:
                    explanation_parts.append(f"{feature} increases risk")
                else:
                    explanation_parts.append(f"{feature} decreases risk")
        
        explanation = ". ".join(explanation_parts) + "."
        
        return {
            "method": "SHAP (Demo)",
            "feature_importance": dict(sorted_features),
            "explanation": explanation,
            "shap_values": list(feature_importance.values()),
            "base_value": 0.0
        }

    def _generate_demo_lime_explanation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo LIME explanation"""
        # Create similar but slightly different feature importance for LIME
        feature_importance = {}
        
        # Speed limit effect (slightly different from SHAP)
        speed_limit = input_data.get('speed_limit', 30)
        feature_importance['speed_limit'] = (speed_limit - 30) / 120
        
        # Weather effect
        weather = input_data.get('weather_conditions', 'Fine no high winds')
        if 'rain' in weather.lower() or 'snow' in weather.lower():
            feature_importance['weather_conditions'] = 0.12
        elif 'fog' in weather.lower():
            feature_importance['weather_conditions'] = 0.20
        else:
            feature_importance['weather_conditions'] = -0.04
        
        # Light conditions effect
        light = input_data.get('light_conditions', 'Daylight')
        if 'dark' in light.lower():
            feature_importance['light_conditions'] = 0.10
        else:
            feature_importance['light_conditions'] = -0.06
        
        # Add some random variation for other features
        for feature in self.feature_names:
            if feature not in feature_importance:
                feature_importance[feature] = np.random.uniform(-0.08, 0.08)
        
        # Sort by absolute importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Generate explanation text
        explanation_parts = []
        for feature, value in sorted_features[:3]:
            if abs(value) > 0.04:
                if value > 0:
                    explanation_parts.append(f"{feature} contributes positively")
                else:
                    explanation_parts.append(f"{feature} contributes negatively")
        
        explanation_text = ". ".join(explanation_parts) + "."
        
        return {
            "method": "LIME (Demo)",
            "feature_importance": dict(sorted_features),
            "explanation": explanation_text,
            "local_prediction": [0.3, 0.5, 0.2],
            "score": 0.85
        }

    def _calculate_agreement(self, shap_explanation: Dict[str, Any], lime_explanation: Dict[str, Any]) -> str:
        """Calculate agreement level between SHAP and LIME explanations"""
        try:
            shap_features = set(shap_explanation.get('feature_importance', {}).keys())
            lime_features = set(lime_explanation.get('feature_importance', {}).keys())
            
            # Calculate overlap in top features
            overlap = len(shap_features.intersection(lime_features))
            total = len(shap_features.union(lime_features))
            
            if total == 0:
                return "no_overlap"
            
            agreement_ratio = overlap / total
            
            if agreement_ratio > 0.7:
                return "high_agreement"
            elif agreement_ratio > 0.4:
                return "moderate_agreement"
            else:
                return "low_agreement"
                
        except Exception:
            return "unknown" 