import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging
import io
import base64

logger = logging.getLogger(__name__)

class VisualizationService:
    """Service for creating data visualizations"""
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        self.colors = {
            'Slight': '#10b981',    # Green
            'Serious': '#f59e0b',   # Orange  
            'Fatal': '#ef4444',     # Red
            'primary': '#2563eb',   # Blue
            'secondary': '#6b7280'  # Gray
        }

    async def create_severity_chart(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create severity distribution chart data"""
        try:
            if not data:
                return self._empty_chart_response("No data available")
            
            # Prepare data for chart
            labels = [item['severity'] for item in data]
            values = [item['count'] for item in data]
            colors = [self.colors.get(label, self.colors['secondary']) for label in labels]
            
            # Create Plotly pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    textinfo='label+percent+value',
                    textposition='auto',
                    hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Accident Severity Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#1f2937'}
                },
                showlegend=True,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151')
            )
            
            chart_json = json.loads(fig.to_json())
            
            return {
                "chart_type": "pie",
                "title": "Accident Severity Distribution",
                "data": chart_json,
                "summary": {
                    "total_predictions": sum(values),
                    "most_common": labels[values.index(max(values))] if values else "None"
                }
            }
            
        except Exception as e:
            logger.error(f"Severity chart creation failed: {str(e)}")
            return self._empty_chart_response("Chart generation failed")

    async def create_trends_chart(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create prediction trends over time chart"""
        try:
            if not data:
                return self._empty_chart_response("No trend data available")
            
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            
            # Create line chart
            fig = go.Figure()
            
            # Add traces for each severity level
            for severity in ['Slight', 'Serious', 'Fatal']:
                severity_data = df[df['severity'] == severity]
                if not severity_data.empty:
                    fig.add_trace(go.Scatter(
                        x=severity_data['date'],
                        y=severity_data['count'],
                        mode='lines+markers',
                        name=severity,
                        line=dict(color=self.colors.get(severity, self.colors['secondary']), width=3),
                        marker=dict(size=6),
                        hovertemplate=f'<b>{severity}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>'
                    ))
            
            fig.update_layout(
                title={
                    'text': 'Prediction Trends Over Time',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#1f2937'}
                },
                xaxis_title='Date',
                yaxis_title='Number of Predictions',
                height=400,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151'),
                hovermode='x unified'
            )
            
            chart_json = json.loads(fig.to_json())
            
            return {
                "chart_type": "line",
                "title": "Prediction Trends Over Time",
                "data": chart_json,
                "summary": {
                    "date_range": f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}",
                    "total_days": len(df['date'].unique())
                }
            }
            
        except Exception as e:
            logger.error(f"Trends chart creation failed: {str(e)}")
            return self._empty_chart_response("Trends chart generation failed")

    async def create_feature_importance_chart(self, importance_data: Dict[str, float]) -> Dict[str, Any]:
        """Create feature importance horizontal bar chart"""
        try:
            if not importance_data:
                return self._empty_chart_response("No feature importance data available")
            
            # Sort features by importance
            sorted_features = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
            features = [item[0] for item in sorted_features[:10]]  # Top 10 features
            importances = [item[1] for item in sorted_features[:10]]
            
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=features,
                    x=importances,
                    orientation='h',
                    marker=dict(
                        color=importances,
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Importance Score")
                    ),
                    text=[f'{imp:.3f}' for imp in importances],
                    textposition='inside',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Top 10 Feature Importance',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#1f2937'}
                },
                xaxis_title='Importance Score',
                yaxis_title='Features',
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151'),
                margin=dict(l=200)  # More space for feature names
            )
            
            chart_json = json.loads(fig.to_json())
            
            return {
                "chart_type": "horizontal_bar",
                "title": "Feature Importance Analysis",
                "data": chart_json,
                "summary": {
                    "most_important": features[0] if features else "None",
                    "least_important": features[-1] if features else "None",
                    "total_features": len(importance_data)
                }
            }
            
        except Exception as e:
            logger.error(f"Feature importance chart creation failed: {str(e)}")
            return self._empty_chart_response("Feature importance chart generation failed")

    async def create_confidence_distribution_chart(self, confidence_data: List[float]) -> Dict[str, Any]:
        """Create confidence score distribution histogram"""
        try:
            if not confidence_data:
                return self._empty_chart_response("No confidence data available")
            
            # Create histogram
            fig = go.Figure(data=[
                go.Histogram(
                    x=confidence_data,
                    nbinsx=20,
                    marker=dict(
                        color=self.colors['primary'],
                        opacity=0.7,
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='Confidence Range: %{x}<br>Count: %{y}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Prediction Confidence Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#1f2937'}
                },
                xaxis_title='Confidence Score',
                yaxis_title='Frequency',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151')
            )
            
            chart_json = json.loads(fig.to_json())
            
            return {
                "chart_type": "histogram",
                "title": "Confidence Distribution",
                "data": chart_json,
                "summary": {
                    "mean_confidence": np.mean(confidence_data),
                    "std_confidence": np.std(confidence_data),
                    "high_confidence_count": sum(1 for c in confidence_data if c > 0.8)
                }
            }
            
        except Exception as e:
            logger.error(f"Confidence distribution chart creation failed: {str(e)}")
            return self._empty_chart_response("Confidence chart generation failed")

    async def create_risk_matrix_chart(self, risk_data: Dict[str, int]) -> Dict[str, Any]:
        """Create risk level matrix visualization"""
        try:
            if not risk_data:
                return self._empty_chart_response("No risk data available")
            
            risk_levels = ['Very Low', 'Low', 'Medium', 'High', 'Critical']
            counts = [risk_data.get(level, 0) for level in risk_levels]
            colors_list = ['#10b981', '#84cc16', '#f59e0b', '#ef4444', '#7c2d12']
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=risk_levels,
                    y=counts,
                    marker=dict(color=colors_list),
                    text=counts,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title={
                    'text': 'Risk Level Distribution',
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16, 'color': '#1f2937'}
                },
                xaxis_title='Risk Level',
                yaxis_title='Count',
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#374151')
            )
            
            chart_json = json.loads(fig.to_json())
            
            return {
                "chart_type": "bar",
                "title": "Risk Level Distribution",
                "data": chart_json,
                "summary": {
                    "highest_risk_count": max(counts),
                    "total_high_risk": sum(counts[-2:])  # High + Critical
                }
            }
            
        except Exception as e:
            logger.error(f"Risk matrix chart creation failed: {str(e)}")
            return self._empty_chart_response("Risk matrix chart generation failed")

    def _empty_chart_response(self, message: str) -> Dict[str, Any]:
        """Return empty chart response"""
        return {
            "chart_type": "empty",
            "title": "No Data",
            "data": {"layout": {"title": message}},
            "summary": {"message": message}
        }

    async def generate_dashboard_charts(self, analytics_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate all dashboard charts at once"""
        try:
            charts = {}
            
            # Severity distribution
            if analytics_data.get('severity_distribution'):
                severity_data = [
                    {'severity': k.title(), 'count': v} 
                    for k, v in analytics_data['severity_distribution'].items()
                ]
                charts['severity'] = await self.create_severity_chart(severity_data)
            
            # Add more charts as needed
            charts['timestamp'] = pd.Timestamp.now().isoformat()
            
            return charts
            
        except Exception as e:
            logger.error(f"Dashboard charts generation failed: {str(e)}")
            return {"error": str(e)}