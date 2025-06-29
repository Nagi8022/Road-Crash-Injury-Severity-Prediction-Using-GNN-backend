import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Handles generation of visualizations for accident data."""
    
    @staticmethod
    def create_severity_pie_chart(data: List[Dict[str, Any]]) -> bytes:
        """Create a pie chart showing the distribution of accident severities."""
        try:
            df = pd.DataFrame(data)
            severity_counts = df['severity'].value_counts().reset_index()
            severity_counts.columns = ['severity', 'count']
            
            fig = px.pie(
                severity_counts,
                values='count',
                names='severity',
                title='Distribution of Accident Severity',
                color='severity',
                color_discrete_map={
                    'Slight': '#2ecc71',
                    'Serious': '#f39c12',
                    'Fatal': '#e74c3c'
                },
                hole=0.3
            )
            
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
            )
            
            fig.update_layout(
                uniformtext_minsize=12,
                uniformtext_mode='hide',
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            return fig.to_image(format='png', scale=2)
            
        except Exception as e:
            logger.error(f"Error creating severity pie chart: {str(e)}")
            raise

    @staticmethod
    def create_time_series_plot(data: List[Dict[str, Any]], time_column: str = 'timestamp') -> bytes:
        """Create a time series plot of accidents over time."""
        try:
            df = pd.DataFrame(data)
            
            # Ensure timestamp is datetime
            if time_column in df.columns:
                df[time_column] = pd.to_datetime(df[time_column])
                df = df.sort_values(time_column)
                
                # Group by date and severity
                df['date'] = df[time_column].dt.date
                time_series = df.groupby(['date', 'severity']).size().unstack(fill_value=0)
                
                fig = px.line(
                    time_series,
                    title='Accidents Over Time by Severity',
                    labels={'value': 'Number of Accidents', 'date': 'Date'},
                    color_discrete_map={
                        'Slight': '#2ecc71',
                        'Serious': '#f39c12',
                        'Fatal': '#e74c3c'
                    }
                )
                
                fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Number of Accidents',
                    legend_title='Severity',
                    hovermode='x unified'
                )
                
                return fig.to_image(format='png', scale=2)
            return None
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {str(e)}")
            raise

    @staticmethod
    def create_feature_importance_plot(importances: Dict[str, float]) -> bytes:
        """Create a bar chart of feature importances."""
        try:
            if not importances:
                return None
                
            features = list(importances.keys())
            values = list(importances.values())
            
            # Sort features by importance
            sorted_idx = np.argsort(values)[-15:]  # Top 15 features
            features = [features[i] for i in sorted_idx]
            values = [values[i] for i in sorted_idx]
            
            fig = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color='#3498db',
                hovertemplate='%{y}: %{x:.3f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Importances',
                xaxis_title='Importance',
                yaxis_title='Features',
                height=500,
                margin=dict(l=150, r=50, t=80, b=50)
            )
            
            return fig.to_image(format='png', scale=2)
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {str(e)}")
            raise

    @staticmethod
    def create_correlation_heatmap(data: List[Dict[str, Any]], numeric_columns: List[str]) -> bytes:
        """Create a correlation heatmap for numeric features."""
        try:
            df = pd.DataFrame(data)
            
            # Convert numeric columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Calculate correlation matrix
            corr = df[numeric_columns].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                colorbar=dict(title='Correlation'),
                hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title='Feature Correlation Heatmap',
                xaxis_title='Features',
                yaxis_title='Features',
                width=800,
                height=700,
                margin=dict(l=150, r=50, t=100, b=150)
            )
            
            return fig.to_image(format='png', scale=2)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            raise

    @staticmethod
    def create_dashboard(data: List[Dict[str, Any]], importances: Optional[Dict[str, float]] = None) -> bytes:
        """Create a comprehensive dashboard with multiple visualizations."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                specs=[[{"type": "pie"}, {"type": "xy"}],
                       [{"type": "bar"}, {"type": "heatmap"}]],
                subplot_titles=(
                    'Accident Severity Distribution',
                    'Accidents Over Time',
                    'Top Feature Importances',
                    'Feature Correlations'
                )
            )
            
            # Add severity pie chart
            df = pd.DataFrame(data)
            severity_counts = df['severity'].value_counts().reset_index()
            severity_counts.columns = ['severity', 'count']
            
            fig.add_trace(
                go.Pie(
                    labels=severity_counts['severity'],
                    values=severity_counts['count'],
                    name='Severity',
                    marker=dict(
                        colors=['#2ecc71', '#f39c12', '#e74c3c'],
                        line=dict(color='#fff', width=2)
                    ),
                    hole=0.4,
                    showlegend=False
                ),
                row=1, col=1
            )
            
            # Add time series if timestamp exists
            if 'timestamp' in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.date
                time_series = df.groupby(['date', 'severity']).size().unstack(fill_value=0)
                
                for severity in time_series.columns:
                    color = {
                        'Slight': '#2ecc71',
                        'Serious': '#f39c12',
                        'Fatal': '#e74c3c'
                    }.get(severity, '#3498db')
                    
                    fig.add_trace(
                        go.Scatter(
                            x=time_series.index,
                            y=time_series[severity],
                            name=severity,
                            mode='lines+markers',
                            line=dict(color=color, width=2),
                            showlegend=False
                        ),
                        row=1, col=2
                    )
            
            # Add feature importances if provided
            if importances:
                features = list(importances.keys())
                values = list(importances.values())
                sorted_idx = np.argsort(values)[-10:]  # Top 10 features
                
                fig.add_trace(
                    go.Bar(
                        x=[values[i] for i in sorted_idx],
                        y=[features[i] for i in sorted_idx],
                        orientation='h',
                        marker_color='#3498db',
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title_text='Accident Analysis Dashboard',
                height=1000,
                width=1200,
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='right',
                    x=1
                )
            )
            
            # Update subplot titles
            for i in range(1, 5):
                fig.update_annotations(font_size=12, selector=dict(text=f'subplot{i}'))
            
            return fig.to_image(format='png', scale=2)
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {str(e)}")
            raise
