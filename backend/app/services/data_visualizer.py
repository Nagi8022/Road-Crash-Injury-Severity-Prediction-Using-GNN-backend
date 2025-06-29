import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import base64
from io import BytesIO
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataVisualizer:
    """Handles generation of visualizations and reports for accident data"""
    
    def __init__(self, output_dir: str = "reports"):
        """Initialize with output directory for saving reports"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set(style="whitegrid")
        
        # Set color palette for severity levels
        self.severity_palette = {
            'FATAL': '#ff4d4d',
            'SERIOUS': '#ff9999',
            'SLIGHT': '#ffcccc'
        }
    
    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive summary statistics"""
        try:
            # Basic statistics
            stats = {
                'total_records': len(df),
                'start_date': df.get('date', pd.Series([pd.NaT])).min(),
                'end_date': df.get('date', pd.Series([pd.NaT])).max(),
                'numeric_stats': {},
                'categorical_stats': {}
            }
            
            # Numeric columns statistics
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_cols:
                stats['numeric_stats'][col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'std': df[col].std(),
                    'missing': df[col].isna().sum()
                }
            
            # Categorical columns statistics
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in cat_cols:
                value_counts = df[col].value_counts().to_dict()
                stats['categorical_stats'][col] = {
                    'unique_values': len(value_counts),
                    'top_values': dict(list(value_counts.items())[:5]),  # Top 5 values
                    'missing': df[col].isna().sum()
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {str(e)}")
            raise
    
    def plot_severity_distribution(self, df: pd.DataFrame) -> str:
        """Generate and return base64 encoded image of severity distribution"""
        try:
            plt.figure(figsize=(10, 6))
            if 'accident_severity' in df.columns:
                ax = sns.countplot(
                    data=df,
                    x='accident_severity',
                    order=['FATAL', 'SERIOUS', 'SLIGHT'],
                    palette=self.severity_palette
                )
                plt.title('Accident Severity Distribution')
                plt.xlabel('Severity')
                plt.ylabel('Count')
                
                # Add count labels on top of bars
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                               (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', 
                               xytext=(0, 10), 
                               textcoords='offset points')
                
                # Save to bytes
                img = BytesIO()
                plt.tight_layout()
                plt.savefig(img, format='png', dpi=100)
                plt.close()
                
                return base64.b64encode(img.getvalue()).decode('utf-8')
            return ""
            
        except Exception as e:
            logger.error(f"Error generating severity distribution plot: {str(e)}")
            return ""
    
    def plot_trend_over_time(self, df: pd.DataFrame, time_col: str = 'date') -> str:
        """Generate time series trend plot"""
        try:
            if time_col in df.columns:
                # Ensure datetime type
                df[time_col] = pd.to_datetime(df[time_col])
                df = df.sort_values(time_col)
                
                # Resample to weekly counts
                weekly = df.set_index(time_col).resample('W').size()
                
                plt.figure(figsize=(12, 6))
                weekly.plot(linewidth=2)
                plt.title('Weekly Accident Trend')
                plt.xlabel('Date')
                plt.ylabel('Number of Accidents')
                plt.grid(True, alpha=0.3)
                
                # Save to bytes
                img = BytesIO()
                plt.tight_layout()
                plt.savefig(img, format='png', dpi=100)
                plt.close()
                
                return base64.b64encode(img.getvalue()).decode('utf-8')
            return ""
            
        except Exception as e:
            logger.error(f"Error generating trend plot: {str(e)}")
            return ""
    
    def plot_feature_distribution(self, df: pd.DataFrame, feature: str) -> str:
        """Generate distribution plot for a specific feature"""
        try:
            if feature in df.columns:
                plt.figure(figsize=(10, 6))
                
                if df[feature].dtype in ['int64', 'float64']:
                    # For numeric features
                    sns.histplot(data=df, x=feature, bins=20, kde=True)
                else:
                    # For categorical features
                    value_counts = df[feature].value_counts().nlargest(10)  # Top 10 categories
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.xticks(rotation=45, ha='right')
                
                plt.title(f'Distribution of {feature}')
                plt.tight_layout()
                
                # Save to bytes
                img = BytesIO()
                plt.savefig(img, format='png', dpi=100)
                plt.close()
                
                return base64.b64encode(img.getvalue()).decode('utf-8')
            return ""
            
        except Exception as e:
            logger.error(f"Error generating {feature} distribution plot: {str(e)}")
            return ""
    
    def generate_html_report(self, df: pd.DataFrame, title: str = "Accident Data Report") -> str:
        """Generate a complete HTML report with visualizations"""
        try:
            # Generate all visualizations
            severity_plot = self.plot_severity_distribution(df)
            trend_plot = self.plot_trend_over_time(df)
            
            # Get important features for visualization
            important_features = ['speed_limit', 'road_type', 'light_conditions', 'weather_conditions']
            feature_plots = {}
            for feature in important_features:
                if feature in df.columns:
                    feature_plots[feature] = self.plot_feature_distribution(df, feature)
            
            # Generate summary statistics
            stats = self.generate_summary_statistics(df)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .section {{ margin-bottom: 30px; }}
                    .plot {{ margin: 20px 0; text-align: center; }}
                    .stats-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                    .stats-table th, .stats-table td {{ 
                        border: 1px solid #ddd; 
                        padding: 8px; 
                        text-align: left; 
                    }}
                    .stats-table th {{ background-color: #f2f2f2; }}
                    .feature-grid {{ 
                        display: grid; 
                        grid-template-columns: repeat(2, 1fr); 
                        gap: 20px; 
                        margin: 20px 0;
                    }}
                    .feature-plot {{ 
                        border: 1px solid #eee; 
                        padding: 15px; 
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{title}</h1>
                    <p>Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Total records: {len(df):,}</p>
                </div>
                
                <div class="section">
                    <h2>1. Summary Statistics</h2>
                    <h3>Numeric Columns</h3>
                    <table class="stats-table">
                        <tr>
                            <th>Column</th>
                            <th>Mean</th>
                            <th>Median</th>
                            <th>Min</th>
                            <th>Max</th>
                            <th>Missing</th>
                        </tr>
            """
            
            # Add numeric stats to HTML
            for col, col_stats in stats.get('numeric_stats', {}).items():
                html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{col_stats['mean']:.2f}</td>
                        <td>{col_stats['median']:.2f}</td>
                        <td>{col_stats['min']}</td>
                        <td>{col_stats['max']}</td>
                        <td>{col_stats['missing']}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                    
                    <h3>Categorical Columns</h3>
                    <table class="stats-table">
                        <tr>
                            <th>Column</th>
                            <th>Unique Values</th>
                            <th>Top Values</th>
                            <th>Missing</th>
                        </tr>
            """
            
            # Add categorical stats to HTML
            for col, col_stats in stats.get('categorical_stats', {}).items():
                top_values = ", ".join([f"{k} ({v})" for k, v in col_stats['top_values'].items()])
                html_content += f"""
                    <tr>
                        <td>{col}</td>
                        <td>{col_stats['unique_values']}</td>
                        <td>{top_values}</td>
                        <td>{col_stats['missing']}</td>
                    </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>2. Key Visualizations</h2>
                    
                    <div class="plot">
                        <h3>Accident Severity Distribution</h3>
                        <img src="data:image/png;base64,""" + severity_plot + """" 
                             alt="Severity Distribution" style="max-width: 80%;">
                    </div>
            """
            
            if trend_plot:
                html_content += f"""
                    <div class="plot">
                        <h3>Accident Trend Over Time</h3>
                        <img src="data:image/png;base64,{trend_plot}" 
                             alt="Accident Trend" style="max-width: 80%;">
                    </div>
                """
            
            # Add feature distributions
            if feature_plots:
                html_content += """
                    <div class="section">
                        <h3>Feature Distributions</h3>
                        <div class="feature-grid">
                """
                
                for feature, plot in feature_plots.items():
                    if plot:
                        html_content += f"""
                            <div class="feature-plot">
                                <h4>{feature.title().replace('_', ' ')}</h4>
                                <img src="data:image/png;base64,{plot}" 
                                     alt="{feature} Distribution" style="max-width: 100%;">
                            </div>
                        """
                
                html_content += """
                        </div>
                    </div>
                """
            
            # Add footer
            html_content += """
                <div class="footer" style="margin-top: 50px; text-align: center; color: #666; font-size: 0.9em;">
                    <p>Report generated by Road Safety Analytics System</p>
                </div>
            </body>
            </html>
            """
            
            # Save the report
            report_path = os.path.join(self.output_dir, f"accident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            raise
    
    def export_to_pdf(self, html_content: str, output_path: Optional[str] = None) -> str:
        """Export HTML content to PDF using WeasyPrint"""
        try:
            from weasyprint import HTML
            
            if not output_path:
                output_path = os.path.join(
                    self.output_dir, 
                    f"accident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                )
            
            HTML(string=html_content).write_pdf(output_path)
            return output_path
            
        except ImportError:
            logger.warning("WeasyPrint not installed. Install with: pip install weasyprint")
            return ""
        except Exception as e:
            logger.error(f"Error exporting to PDF: {str(e)}")
            return ""

# Example usage
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    data = {
        'date': np.random.choice(dates, 1000),
        'accident_severity': np.random.choice(['FATAL', 'SERIOUS', 'SLIGHT'], 1000, p=[0.05, 0.25, 0.7]),
        'speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], 1000, p=[0.1, 0.3, 0.2, 0.2, 0.15, 0.05]),
        'road_type': np.random.choice(['Single carriageway', 'Dual carriageway', 'Motorway', 'Slip road'], 1000, p=[0.5, 0.3, 0.15, 0.05]),
        'light_conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - no lighting'], 1000, p=[0.6, 0.3, 0.1]),
        'weather_conditions': np.random.choice(['Fine no high winds', 'Raining without high winds', 'Raining with high winds', 'Fog or mist'], 1000, p=[0.6, 0.2, 0.1, 0.1]),
        'road_surface_conditions': np.random.choice(['Dry', 'Wet or damp', 'Snow', 'Frost or ice'], 1000, p=[0.7, 0.2, 0.05, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Create visualizer instance
    visualizer = DataVisualizer()
    
    # Generate HTML report
    report_path = visualizer.generate_html_report(df, "Sample Accident Data Report")
    print(f"HTML report generated: {report_path}")
    
    # Optional: Export to PDF (requires WeasyPrint)
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        pdf_path = visualizer.export_to_pdf(html_content)
        if pdf_path:
            print(f"PDF report generated: {pdf_path}")
    except Exception as e:
        print(f"Could not generate PDF: {str(e)}")
