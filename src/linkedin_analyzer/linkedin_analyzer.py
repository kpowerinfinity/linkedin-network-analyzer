import click
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import track
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import networkx as nx
from fpdf import FPDF
import csv

console = Console()

@dataclass
class ConnectionInsight:
    name: str
    company: str
    industry: str
    last_contact: datetime
    priority_score: float
    recommended_action: str
    common_interests: List[str]

class LinkedInAnalyzer:
    def __init__(self, config_path: str, output_dir: str = None, profile_path: str = None):
        # Load config first
        self.config = self._load_config(config_path)
        
        
        # Get the directory where the script is located
        script_dir = Path(__file__).parent.absolute()
        
        # Create output directory using CLI parameter if provided, otherwise use default
        self.output_path = Path(output_dir) if output_dir else script_dir / "output"
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        # Create visualizations subdirectory if enabled
        self.visualizations_path = self.output_path / "visualizations"
        if self.config.get('generate_visualizations', False):
            self.visualizations_path.mkdir(exist_ok=True)
        
        # Setup logging last, after output_path is created
        self.setup_logging()
        # Load profile data
        self.profile_path = profile_path or "Profile.csv"
        self.full_name = self._load_profile_data()
        


    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def setup_logging(self):
        # Update log file path to be in output directory
        log_file = self.output_path / self.config.get('log_file', 'linkedin_analyzer.log')
        logging.basicConfig(
            level=self.config.get('logging_level', 'INFO'),
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=log_file
        )

    def _parse_linkedin_date(self, date_str: str) -> datetime:
        """Handle various LinkedIn date formats."""
        try:
            # Try different date formats
            formats = [
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%d %b %Y',
                '%B %d, %Y',
                '%Y-%m-%d %H:%M:%S %Z'
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {date_str}")
        except Exception as e:
            logging.error(f"Date parsing error: {str(e)}")
            return None

    def load_connections(self, connections_path: str) -> pd.DataFrame:
        """Enhanced connection loading with data cleaning."""
        
        logging.info("Loading connections from: %s", connections_path)
        try:
            df = pd.read_csv(connections_path, skiprows=2)  # Skip the notes rows
            logging.info("Successfully loaded CSV file")
            
            # Log the first few rows
            logging.info("First 3 rows of data:")
            for idx, row in df.head(3).iterrows():
                logging.info("Row %d: %s", idx, row.to_dict())
            
            # Log column names
            logging.info("Columns in dataframe: %s", df.columns.tolist())
            
            # Log basic stats
            logging.info("DataFrame shape: %s", df.shape)
            logging.info("Number of null values:\n%s", df.isnull().sum())
            
            #return df
            
        except pd.errors.EmptyDataError:
            logging.error("CSV file is empty")
            raise
        except pd.errors.ParserError as e:
            logging.error("Error parsing CSV: %s", str(e))
            raise
        except Exception as e:
            logging.error("Unexpected error loading connections: %s", str(e))
            raise
        
        
        # Standardize column names
        standard_columns = {
            'First Name': 'first_name',
            'Last Name': 'last_name',
            'Company': 'company',
            'Position': 'position',
            'Connected On': 'connected_on',
            'Industry': 'industry'
        }
        
        df = df.rename(columns={k: v for k, v in standard_columns.items() if k in df.columns})
        
        for idx, row in df.head(3).iterrows():
            logging.info("Row %d: %s", idx, row.to_dict())

        # Clean and transform data
        df['connected_on'] = pd.to_datetime(df['connected_on'].apply(self._parse_linkedin_date))
        df['full_name'] = df['first_name'] + ' ' + df['last_name']
        

        # Initialize industry column if it doesn't exist
        if 'industry' not in df.columns:
            df['industry'] = 'Unknown'
        else:
            # Extract and standardize industries
            df['industry'] = df['industry'].fillna('Unknown')
            df['industry'] = df['industry'].str.title()


        for idx, row in df.head(3).iterrows():
            logging.info("Row %d: %s", idx, row.to_dict())

        # Log summary statistics of processed data
        logging.info("Data summary after processing:")
        logging.info("Date range: %s to %s", 
                    df['connected_on'].min().strftime('%Y-%m-%d'),
                    df['connected_on'].max().strftime('%Y-%m-%d'))
        

        for idx, row in df.head(3).iterrows():
            logging.info("Row %d: %s", idx, row.to_dict())
        
        company_stats = df['company'].value_counts()
        logging.info("Top 5 companies:")
        for company, count in company_stats.head().items():
            logging.info("  %s: %d connections", company, count)
            
        if 'industry' in df.columns:
            industry_stats = df['industry'].value_counts()
            logging.info("Top 5 industries:")
            for industry, count in industry_stats.head().items():
                logging.info("  %s: %d connections", industry, count)
        
        logging.info("Connection growth by year:")
        yearly_stats = df.groupby(df['connected_on'].dt.year).size()
        for year, count in yearly_stats.items():
            logging.info("  %d: %d new connections", year, count)
        
        return df

    def load_messages(self, messages_path: str) -> pd.DataFrame:
        """Enhanced message loading with conversation analysis."""
        df = pd.read_csv(messages_path)
        df['DATE'] = pd.to_datetime(df['DATE'].apply(self._parse_linkedin_date))
        
        # Add message analysis
        df['message_length'] = df['CONTENT'].str.len()
        df['has_question'] = df['CONTENT'].str.contains(r'\?', regex=True)
        df['sentiment'] = self._analyze_sentiment(df['CONTENT'])
        # Log first 3 rows of messages data
        for idx, row in df.head(3).iterrows():
            logging.info("Message %d: %s", idx, row.to_dict())
            
        # Log summary statistics
        logging.info("Messages data summary:")
        logging.info("Date range: %s to %s",
                    df['DATE'].min().strftime('%Y-%m-%d'),
                    df['DATE'].max().strftime('%Y-%m-%d'))
        logging.info("Total messages: %d", len(df))
        
        # Message statistics
        logging.info("Message length statistics:")
        logging.info("  Average length: %.1f characters", df['message_length'].mean())
        logging.info("  Max length: %d characters", df['message_length'].max())
        
        return df

    def _analyze_sentiment(self, texts: pd.Series) -> pd.Series:
        """Basic sentiment analysis using keyword matching."""
        positive_words = set(['thank', 'thanks', 'appreciate', 'great', 'good', 'excellent'])
        negative_words = set(['sorry', 'issue', 'problem', 'concern', 'wrong'])
        
        def get_sentiment(text):
            if pd.isna(text):
                return 0
            text = text.lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            return pos_count - neg_count
        
        return texts.apply(get_sentiment)

    def analyze_network_growth(self, connections_df: pd.DataFrame, messages_df: Optional[pd.DataFrame] = None) -> Dict:
        """Enhanced network growth analysis."""
        growth_analysis = {
            'total_connections': len(connections_df),
            'growth_by_time': self._analyze_growth_patterns(connections_df),
            'industry_distribution': self._analyze_industry_distribution(connections_df),
            'company_distribution': self._analyze_company_distribution(connections_df),
            'connection_quality': self._analyze_connection_quality(connections_df)
        }
        
        # Add message-based reconnection suggestions if messages data is provided
        if messages_df is not None:
            growth_analysis['message_based_reconnections'] = self.generate_message_based_reconnections(
                connections_df, messages_df
            )
        
        # Generate growth visualization
        self._create_growth_visualization(connections_df)
        
        return growth_analysis
    

    def _analyze_industry_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze the distribution of industries in the network."""
        industry_counts = df['industry'].value_counts()
        total_connections = len(df)
        
        return {
            'top_industries': industry_counts.head(5).to_dict(),
            'industry_percentages': (industry_counts / total_connections * 100).round(2).head(5).to_dict(),
            'total_industries': len(industry_counts),
            'diversity_score': (1 - (industry_counts / total_connections).pow(2).sum()).round(3),  # Herfindahl-Hirschman Index inverse
            'unknown_industry_count': int(industry_counts.get('Unknown', 0)),
            'unknown_industry_percentage': round((industry_counts.get('Unknown', 0) / total_connections * 100), 2)
        }


    def _analyze_company_distribution(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the distribution of companies in the network.
        
        Args:
            df: DataFrame containing connection data with 'company' column
            
        Returns:
            Dict containing company distribution metrics
        """
        company_counts = df['company'].value_counts()
        total_connections = len(df)
        
        # Calculate company size tiers
        company_tiers = {
            'large': company_counts[company_counts >= 10].count(),
            'medium': company_counts[(company_counts >= 5) & (company_counts < 10)].count(),
            'small': company_counts[company_counts < 5].count()
        }
        
        return {
            'top_companies': company_counts.head(10).to_dict(),
            'company_percentages': (company_counts / total_connections * 100).round(2).head(5).to_dict(),
            'total_companies': len(company_counts),
            'company_tiers': company_tiers,
            'diversity_score': (1 - (company_counts / total_connections).pow(2).sum()).round(3),
            'unknown_company_count': int(company_counts.get('Unknown', 0)),
            'unknown_company_percentage': round((company_counts.get('Unknown', 0) / total_connections * 100), 2)
        }

    def _analyze_connection_quality(self, df: pd.DataFrame) -> Dict:
        """
        Analyze the quality of connections based on various metrics.
        
        Args:
            df: DataFrame containing connection data
            
        Returns:
            Dict containing connection quality metrics
        """
        # Calculate connection age
        current_date = datetime.now()
        df['connection_age'] = (current_date - df['connected_on']).dt.days
        
        # Define recent connections (last 90 days)
        recent_mask = df['connection_age'] <= 90
        
        # Calculate industry overlap (connections in same industry)
        industry_counts = df['industry'].value_counts()
        industry_overlap = (industry_counts / len(df) * 100).round(2)
        
        # Calculate company overlap (connections in same company)
        company_counts = df['company'].value_counts()
        company_overlap = (company_counts / len(df) * 100).round(2)
        
        # Calculate network freshness score (weighted average of recent connections)
        freshness_score = (
            (len(df[recent_mask]) / len(df)) * 0.7 +  # 70% weight to recent connections
            (1 - df['connection_age'].mean() / df['connection_age'].max()) * 0.3  # 30% weight to overall age distribution
        ).round(3)
        
        return {
            'total_connections': len(df),
            'recent_connections': len(df[recent_mask]),
            'avg_connection_age_days': int(df['connection_age'].mean()),
            'network_freshness_score': freshness_score,
            'connection_distribution': {
                'last_30_days': len(df[df['connection_age'] <= 30]),
                'last_90_days': len(df[df['connection_age'] <= 90]),
                'last_180_days': len(df[df['connection_age'] <= 180]),
                'last_365_days': len(df[df['connection_age'] <= 365]),
            },
            'top_overlapping_industries': industry_overlap.head(5).to_dict(),
            'top_overlapping_companies': company_overlap.head(5).to_dict(),
            'network_density': {
                'industry_concentration': (industry_counts.max() / len(df)).round(3),
                'company_concentration': (company_counts.max() / len(df)).round(3)
            }
        }

    def _analyze_growth_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze detailed growth patterns."""
        monthly_growth = df.groupby(pd.Grouper(key='connected_on', freq='ME')).size()
        
        growth_patterns = {
            'monthly_growth': monthly_growth.to_dict(),
            'avg_monthly_growth': monthly_growth.mean(),
            'growth_trend': self._calculate_growth_trend(monthly_growth),
            'seasonal_patterns': self._identify_seasonal_patterns(monthly_growth)
        }
        
        return growth_patterns

    def _calculate_growth_trend(self, monthly_growth: pd.Series) -> str:
        """Calculate the overall growth trend."""
        trend = np.polyfit(range(len(monthly_growth)), monthly_growth.values, 1)[0]
        if trend > 0.5:
            return "Strong growth"
        elif trend > 0:
            return "Moderate growth"
        else:
            return "Declining growth"

    def _identify_seasonal_patterns(self, monthly_growth: pd.Series) -> Dict:
        """Identify seasonal patterns in network growth."""
        monthly_growth.index = monthly_growth.index.month
        monthly_avg = monthly_growth.groupby(level=0).mean()
        
        return {
            'highest_growth_month': monthly_avg.idxmax(),
            'lowest_growth_month': monthly_avg.idxmin(),
            'monthly_averages': monthly_avg.to_dict()
        }

    def generate_visualizations(self, connections_df: pd.DataFrame, messages_df: pd.DataFrame):
        """Generate comprehensive network visualizations."""
        # Network growth over time
        plt.figure(figsize=(12, 6))
        connections_df['connected_on'].value_counts().sort_index().cumsum().plot()
        plt.title('Network Growth Over Time')
        plt.savefig(self.visualizations_path / 'network_growth.png')
        plt.close()

        # Industry distribution
        plt.figure(figsize=(12, 6))
        connections_df['industry'].value_counts().head(10).plot(kind='bar')
        plt.title('Top 10 Industries in Network')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.visualizations_path / 'industry_distribution.png')
        plt.close()

        # Message activity heatmap
        messages_df['hour'] = messages_df['DATE'].dt.hour
        messages_df['day'] = messages_df['DATE'].dt.day_name()
        activity = pd.crosstab(messages_df['day'], messages_df['hour'])
        plt.figure(figsize=(15, 8))
        sns.heatmap(activity, cmap='YlOrRd')
        plt.title('Message Activity Heatmap')
        plt.savefig(self.visualizations_path / 'message_activity.png')
        plt.close()

        # Network graph
        self._create_network_graph(connections_df, messages_df)

    def _create_network_graph(self, connections_df: pd.DataFrame, messages_df: pd.DataFrame):
        """Create a network graph visualization."""
        G = nx.Graph()
        
        # Add nodes for connections
        for _, row in connections_df.iterrows():
            G.add_node(row['full_name'], 
                      industry=row['industry'],
                      company=row['company'])
        
        # Add edges for messages
        message_counts = messages_df.groupby(['FROM', 'TO']).size()
        for (sender, recipient), count in message_counts.items():
            if sender in G and recipient in G:
                G.add_edge(sender, recipient, weight=count)
        
        # Create visualization
        plt.figure(figsize=(15, 15))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, 
                node_color='lightblue',
                node_size=50,
                with_labels=False)
        plt.savefig(self.visualizations_path / 'network_graph.png')
        plt.close()

    def export_insights(self, insights: Dict, output_path: str, format: str = 'all'):
        """Export insights in multiple formats."""
        # Convert output_path to be within output directory
        base_path = self.output_path / Path(output_path).stem
        
        if format in ['json', 'all']:
            self._export_json(insights, f"{base_path}.json")
        
        if format in ['csv', 'all']:
            self._export_csv(insights, f"{base_path}.csv")
        
        if format in ['pdf', 'all']:
            self._export_pdf(insights, f"{base_path}.pdf")
        
        if format in ['html', 'all']:
            self._export_html(insights, f"{base_path}.html")

    def _export_json(self, insights: Dict, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(insights, f, indent=2, default=str)

    def _export_csv(self, insights: Dict, filepath: str):
        """Export main metrics to CSV."""
        if 'reconnection_plan' in insights:
            df = pd.DataFrame(insights['reconnection_plan'])
            df.to_csv(filepath, index=False)

    def _export_pdf(self, insights: Dict, filepath: str):
        """Create a formatted PDF report."""
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'LinkedIn Network Analysis Report', 0, 1, 'C')
        
        # Add sections
        pdf.set_font('Arial', '', 12)
        for section, data in insights.items():
            if isinstance(data, dict):
                pdf.cell(0, 10, f'\n{section.replace("_", " ").title()}:', 0, 1)
                for key, value in data.items():
                    pdf.cell(0, 10, f'{key}: {value}', 0, 1)
        
        pdf.output(filepath)

    def _export_html(self, insights: Dict, filepath: str):
        """Create an interactive HTML report."""
        html_content = f"""
        <html>
        <head>
            <title>LinkedIn Network Analysis</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        </head>
        <body class="p-8">
            <h1 class="text-3xl font-bold mb-6">LinkedIn Network Analysis</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                {self._generate_html_cards(insights)}
            </div>
            
            <div class="mt-8">
                <h2 class="text-2xl font-bold mb-4">Visualizations</h2>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <img src="visualizations/network_growth.png" alt="Network Growth">
                    <img src="visualizations/industry_distribution.png" alt="Industry Distribution">
                    <img src="visualizations/message_activity.png" alt="Message Activity">
                    <img src="visualizations/network_graph.png" alt="Network Graph">
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)

    def _generate_html_cards(self, insights: Dict) -> str:
        """Generate HTML cards for each insight section."""
        cards = []
        for section, data in insights.items():
            if isinstance(data, dict):
                card_content = f"""
                <div class="bg-white rounded-lg shadow-lg p-6">
                    <h3 class="text-xl font-bold mb-4">{section.replace('_', ' ').title()}</h3>
                    <ul class="space-y-2">
                        {self._generate_html_list_items(data)}
                    </ul>
                </div>
                """
                cards.append(card_content)
        return '\n'.join(cards)

    def _generate_html_list_items(self, data: Dict) -> str:
        """Generate HTML list items for insight data."""
        items = []
        for key, value in data.items():
            if isinstance(value, (int, float, str)):
                items.append(f'<li><span class="font-semibold">{key}:</span> {value}</li>')
        return '\n'.join(items)

    def _create_growth_visualization(self, connections_df: pd.DataFrame):
        """Create an interactive growth visualization using plotly."""
        try:
            # Ensure visualizations directory exists
            self.visualizations_path.mkdir(exist_ok=True, parents=True)
            
            # Create daily connection counts
            daily_counts = connections_df['connected_on'].value_counts().sort_index()
            cumulative_growth = daily_counts.cumsum()
            
            # Create the main growth figure
            fig = go.Figure()
            
            # Add cumulative growth line
            fig.add_trace(go.Scatter(
                x=cumulative_growth.index,
                y=cumulative_growth.values,
                name='Total Connections',
                line=dict(color='blue', width=2),
                hovertemplate='Date: %{x}<br>Total Connections: %{y}<extra></extra>'
            ))
            
            # Add daily growth bars
            fig.add_trace(go.Bar(
                x=daily_counts.index,
                y=daily_counts.values,
                name='Daily New Connections',
                marker_color='lightblue',
                opacity=0.5,
                hovertemplate='Date: %{x}<br>New Connections: %{y}<extra></extra>'
            ))
            
            # Calculate and add trend line
            x_numeric = np.arange(len(cumulative_growth))
            z = np.polyfit(x_numeric, cumulative_growth.values, 1)
            p = np.poly1d(z)
            
            fig.add_trace(go.Scatter(
                x=cumulative_growth.index,
                y=p(x_numeric),
                name='Growth Trend',
                line=dict(color='red', dash='dash'),
                hovertemplate='Date: %{x}<br>Trend: %{y:.0f}<extra></extra>'
            ))
            
            # Update layout
            fig.update_layout(
                title='Network Growth Over Time',
                xaxis_title='Date',
                yaxis_title='Number of Connections',
                hovermode='x unified',
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            # Save the interactive plot - convert Path to string and ensure directory exists
            interactive_path = self.visualizations_path / 'network_growth_interactive.html'
            fig.write_html(str(interactive_path))
            
            # Save a static version for PDF reports
            static_path = self.visualizations_path / 'network_growth_detailed.png'
            fig.write_image(str(static_path))
            
            logging.info(f"Growth visualizations saved to {self.visualizations_path}")
            
        except Exception as e:
            logging.error(f"Error creating growth visualization: {str(e)}")
            logging.error(f"Attempted to save to: {self.visualizations_path}")
            # Continue execution even if visualization fails
            pass


    
    def _generate_reconnection_suggestions(self, df: pd.DataFrame) -> Dict:
        """Analyze network for reconnection opportunities with senior leaders, founders, and investors."""
        # Define company size thresholds (can be adjusted)
        large_company_list = {
            "Google",
            "Microsoft",
            "Apple",
            "Amazon",
            "Meta",
            "IBM",
            "Intel Corporation",
            "Cisco",
            "Oracle",
            "Salesforce",
            "Adobe",
            "SAP",
            "Uber",
            "Airbnb",
            "Netflix",
            "Spotify",
            "Dropbox",
            "Slack",
            "Zoom Video Communications",
            "Palantir Technologies",
            "Stripe",
            "Square",
            "Pinterest",
            "Snap Inc.",
            "Twitter",
            "LinkedIn",
            "Tesla",
            "SpaceX",
            "NVIDIA",
            "AMD",
            "Qualcomm",
            "Dell Technologies",
            "HP",
            "Xerox",
            "Samsung Electronics",
            "Sony",
            "Alibaba Group",
            "Tencent",
            "ByteDance",
            "Shopify",
            "Atlassian",
            "GitHub",
            "Red Hat",
            "VMware",
            "ServiceNow",
            "Workday",
            "Intuit",
            "PayPal",
            "Coinbase",
            "Robinhood"
        }
        
        # Pattern matching for different roles (without capture groups)
        senior_patterns = r'(?:Chief|CEO|CTO|CFO|COO|President|VP|Vice President|Head|Director)'
        founder_patterns = r'(?:Founder|Co-Founder|Founding)'
        investor_patterns = r'(?:Investor|VC|Partner|Angel|Capital|Investment|Ventures|Private Equity|PE|Managing Director)'
        
        # Get max suggestions from config, default to 20 if not specified
        max_suggestions = self.config.get('max_suggestions_per_category', 20)
        
        # Filter connections for each category using str.contains with regex=True
        senior_large_df = df[
            (df['position'].str.contains(senior_patterns, case=False, regex=True, na=False)) & 
            (df['company'].isin(large_company_list))
        ]
        
        senior_small_df = df[
            (df['position'].str.contains(senior_patterns, case=False, regex=True, na=False)) & 
            (~df['company'].isin(large_company_list))
        ]
        
        founders_df = df[
            df['position'].str.contains(founder_patterns, case=False, regex=True, na=False)
        ]
        
        investors_df = df[
            df['position'].str.contains(investor_patterns, case=False, regex=True, na=False)
        ]
        
        # Sample from each category, handling cases where there might be fewer entries than max_suggestions
        return {
            'senior_large': senior_large_df.sample(n=min(max_suggestions, len(senior_large_df))).to_dict('records'),
            'senior_small': senior_small_df.sample(n=min(max_suggestions, len(senior_small_df))).to_dict('records'),
            'founders': founders_df.sample(n=min(max_suggestions, len(founders_df))).to_dict('records'),
            'investors': investors_df.sample(n=min(max_suggestions, len(investors_df))).to_dict('records')
        }

    def export_reconnection_suggestions(self, suggestions: Dict, output_path: str):
        """Export reconnection suggestions to a file."""
        with open(output_path, 'w') as f:
            f.write("LinkedIn Network Reconnection Suggestions\n")
            f.write("=======================================\n\n")
            
            f.write("Senior Leaders at Large Companies\n")
            f.write("--------------------------------\n")
            for contact in suggestions['senior_large']:
                f.write(f"{contact['full_name']} - {contact['position']} at {contact['company']}\n")
            
            f.write("\nSenior Leaders at Growth Companies\n")
            f.write("--------------------------------\n")
            for contact in suggestions['senior_small']:
                f.write(f"{contact['full_name']} - {contact['position']} at {contact['company']}\n")
            
            f.write("\nFounders\n")
            f.write("--------\n")
            for contact in suggestions['founders']:
                f.write(f"{contact['full_name']} - {contact['position']} at {contact['company']}\n")
            
            f.write("\nInvestors\n")
            f.write("---------\n")
            for contact in suggestions['investors']:
                f.write(f"{contact['full_name']} - {contact['position']} at {contact['company']}\n")

    def generate_message_based_reconnections(self, connections_df: pd.DataFrame, messages_df: pd.DataFrame) -> Dict:
        """
        Generate reconnection suggestions based on messaging history and time since last contact.
        Only includes connections with at least 3 messages exchanged.
        """
        logging.info("Starting generate_message_based_reconnections")
        
        # Use the loaded full name instead of getting it from config
        if not self.full_name:
            logging.warning("full_name not available. Message analysis may be inaccurate.")
            return {}

        # Create a column for the other party in each conversation
        try:
            messages_df['other_party'] = messages_df.apply(
                lambda row: row['TO'] if row['FROM'] == self.full_name else row['FROM'], 
                axis=1
            )
            logging.info("Successfully created other_party column")
        except KeyError as e:
            logging.error(f"KeyError when creating other_party column: {str(e)}")
            logging.error(f"Available columns were: {messages_df.columns.tolist()}")
            raise

        # Count total messages per conversation (both sent and received)
        try:
            message_counts = messages_df.groupby('other_party').size().reset_index(name='message_count')
            logging.info("Successfully created message_counts")
        except Exception as e:
            logging.error(f"Error creating message_counts: {str(e)}")
            raise
        
        # Filter for connections with at least 3 messages
        active_connections = message_counts[message_counts['message_count'] >= 3]
        
        # Get the last message date for each connection
        try:
            last_message_dates = (
                messages_df.groupby('other_party')['DATE']
                .max()
                .reset_index()
            )
            logging.info("Successfully created last_message_dates")
        except Exception as e:
            logging.error(f"Error creating last_message_dates: {str(e)}")
            raise

        # Calculate time since last message
        current_date = datetime.now()
        last_message_dates['years_since_contact'] = (
            current_date - last_message_dates['DATE']
        ).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Create time period buckets
        time_periods = {
            '1_year': (0, 1),
            '2_4_years': (2, 4),
            '5_8_years': (5, 8),
            '8_15_years': (8, 15),
            '15_plus_years': (15, float('inf'))
        }
        
        reconnection_suggestions = {}
        
        for period, (min_years, max_years) in time_periods.items():
            period_contacts = last_message_dates[
                (last_message_dates['years_since_contact'] >= min_years) &
                (last_message_dates['years_since_contact'] < max_years)
            ]
            
            # Merge with connections data to get full contact information
            period_contacts = period_contacts.merge(
                connections_df,
                left_on='other_party',
                right_on='full_name',
                how='inner'
            )
            
            # Merge with message counts
            period_contacts = period_contacts.merge(
                active_connections[['other_party', 'message_count']],
                on='other_party',
                how='inner'
            )
            
            # Sort by message count (prioritize connections with more interaction)
            period_contacts = period_contacts.sort_values('message_count', ascending=False)
            
            # Randomly select 10 contacts if there are more than 10
            if len(period_contacts) > 10:
                period_contacts = period_contacts.sample(n=10, random_state=42)  # Using fixed random_state for reproducibility
            
            # Format the suggestions
            suggestions = []
            for _, contact in period_contacts.iterrows():
                suggestion = {
                    'name': contact['other_party'],
                    'company': contact['company'],
                    'position': contact['position'],
                    'last_contact': contact['DATE'].strftime('%Y-%m-%d'),
                    'message_count': int(contact['message_count']),
                    'years_since_contact': round(contact['years_since_contact'], 1)
                }
                suggestions.append(suggestion)
            
            reconnection_suggestions[period] = suggestions
        
        return reconnection_suggestions

    def export_message_based_reconnections(self, message_suggestions: Dict, output_path: str):
        """
        Export message-based reconnection suggestions to a formatted text file.
        
        Args:
            message_suggestions: Dictionary containing reconnection suggestions by time period
            output_path: Path to save the output file
        """
        period_headers = {
            'last_year': 'Connections to Reconnect With (Last Year)',
            'two_to_four_years': 'Connections to Reconnect With (2-4 Years Ago)',
            'five_to_eight_years': 'Connections to Reconnect With (5-8 Years Ago)',
            'eight_to_fifteen_years': 'Connections to Reconnect With (8-15 Years Ago)',
            'over_fifteen_years': 'Connections to Reconnect With (Over 15 Years Ago)'
        }
        
        with open(output_path, 'w') as f:
            f.write("LinkedIn Message-Based Reconnection Suggestions\n")
            f.write("===========================================\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for period, contacts in message_suggestions.items():
                if not contacts:  # Skip empty periods
                    continue
                
                f.write(f"{period_headers.get(period, period)}\n")
                f.write("-" * len(period_headers.get(period, period)) + "\n\n")
                
                for contact in contacts:
                    f.write(f"""{contact['name']} - {contact['position']} at {contact['company']}
Last Contact: {contact['last_contact']}
Messages: {contact['message_count']} | Years Since: {contact['years_since_contact']}\n\n""")
                
                f.write("\n")
            
            # Add summary at the end
            f.write("\nSummary\n")
            f.write("-------\n")
            total_suggestions = sum(len(contacts) for contacts in message_suggestions.values())
            f.write(f"Total reconnection suggestions: {total_suggestions}\n")
            for period in period_headers.keys():
                if period in message_suggestions:
                    f.write(f"{period_headers[period]}: {len(message_suggestions[period])} contacts\n")

    def _load_profile_data(self) -> str:
        """Load profile data from CSV and construct full name."""
        try:
            profile_df = pd.read_csv(self.profile_path)
            if 'First Name' in profile_df.columns and 'Last Name' in profile_df.columns:
                # Get the first row's first and last name
                first_name = profile_df['First Name'].iloc[0]
                last_name = profile_df['Last Name'].iloc[0]
                full_name = f"{first_name} {last_name}".strip()
                logging.info(f"Loaded profile data for: {full_name}")
                return full_name
            else:
                logging.error("Profile CSV missing required columns: 'First Name' and/or 'Last Name'")
                raise ValueError("Profile CSV must contain 'First Name' and 'Last Name' columns")
        except Exception as e:
            logging.error(f"Error loading profile data: {str(e)}")
            raise


@click.group()
def cli():
    """Enhanced LinkedIn Network Analysis CLI"""
    pass

@cli.command()
@click.option('--config', default='config.yml', help='Path to configuration file')
@click.option('--connections', required=True, help='Path to connections export CSV')
@click.option('--messages', required=True, help='Path to messages export CSV')
@click.option('--profile', default='Profile.csv', help='Path to profile CSV containing first and last name')
@click.option('--output', default='linkedin_insights', help='Output file base name')
@click.option('--output-dir', help='Output directory path')
@click.option('--reconnect/--no-reconnect', default=True, help='Generate reconnection suggestions')
@click.option('--format', default='all', type=click.Choice(['json', 'csv', 'pdf', 'html', 'all']))
def analyze(config: str, connections: str, messages: str, profile: str, output: str, output_dir: str, format: str, reconnect: bool):
    """Analyze LinkedIn network and generate comprehensive insights."""
    try:
        analyzer = LinkedInAnalyzer(config, output_dir, profile)
        console.print("[bold green]Loading and processing data...[/bold green]")
        
        with console.status("[bold green]Analyzing network...") as status:
            # Load and process data
            logging.info("Loading connections data at line 892")
            connections_df = analyzer.load_connections(connections)
            logging.info(f"Connections data loaded with columns: {connections_df.columns.tolist()}")
            
            logging.info("Loading messages data at line 896")
            messages_df = analyzer.load_messages(messages)
            logging.info(f"Messages data loaded with columns: {messages_df.columns.tolist()}")
            
            # Generate insights
            network_growth = analyzer.analyze_network_growth(connections_df, messages_df)
            logging.info(f"Network growth analysis complete at line 901")
            
            # Only generate visualizations if enabled in config
            if analyzer.config.get('generate_visualizations', False):
                analyzer.generate_visualizations(connections_df, messages_df)
                logging.info("Visualizations generated at line 906")
            
            # Compile insights
            insights = {
                'network_growth': network_growth,
                'generated_at': datetime.now().isoformat()
            }
            
            # Generate reconnection suggestions if requested
            if reconnect:
                logging.info("Starting reconnection suggestions generation")
                output_file = analyzer.output_path / f"{output}_reconnect.txt"
                try:
                    suggestions = analyzer._generate_reconnection_suggestions(connections_df)
                    message_suggestions = analyzer.generate_message_based_reconnections(connections_df, messages_df)
                    
                    analyzer.export_reconnection_suggestions(suggestions, output_file)
                    analyzer.export_message_based_reconnections(
                        message_suggestions, 
                        analyzer.output_path / f"{output}_message_reconnect.txt"
                    )
                    logging.info("Reconnection suggestions exported")
                except Exception as e:
                    logging.error(f"Error during reconnection suggestions: {str(e)}")
                    raise
                
                console.print(f"[bold green]Reconnection suggestions generated and saved to {output_file}[/bold green]")
            
            # Export results
            analyzer.export_insights(insights, output, format)
        
        console.print(f"\n[bold green]Analysis complete! Results exported to {output}.[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Error: {str(e)}[/bold red]")
        logging.error(f"Error during analysis: {str(e)}")

if __name__ == '__main__':
    cli()
    
