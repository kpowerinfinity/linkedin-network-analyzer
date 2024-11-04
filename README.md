# LinkedIn Network Analyzer

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A powerful command-line tool for analyzing your LinkedIn network data, providing insights into your professional connections, communication patterns, and networking opportunities.

## 🚀 Features

- **Network Growth Analysis**
  - Track connection growth over time
  - Identify seasonal patterns
  - Analyze industry and company distribution

- **Communication Insights**
  - Message frequency analysis
  - Response time patterns
  - Engagement metrics

- **Visual Analytics**
  - Network growth charts
  - Industry distribution graphs
  - Message activity heatmaps
  - Interactive network graphs

- **Reconnection Recommendations**
  - Identify stale connections
  - Prioritized outreach suggestions
  - Industry-based targeting

- **Multiple Export Formats**
  - JSON for data processing
  - CSV for spreadsheet analysis
  - PDF reports
  - Interactive HTML dashboards

## 📋 Prerequisites

- Python 3.8+
- LinkedIn data export files (Connections.csv and Messages.csv)
- Required system libraries for visualization support

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/linkedin-network-analyzer.git
cd linkedin-network-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate

# Install the package
pip install -e .
```

## 💻 Usage

1. **Export your LinkedIn data**
   - Go to LinkedIn > Settings & Privacy > Get a copy of your data
   - Select "Connections" and "Messages"
   - Download and extract the CSV files

2. **Configure the analyzer**
   ```yaml
   # config/default_config.yml
   logging_level: INFO
   stale_connection_threshold_days: 180
   priority_industries:
     - Software Development
     - Artificial Intelligence
   priority_companies:
     - Google
     - Microsoft
   ```

3. **Run the analyzer**
   ```bash
   linkedin-analyzer analyze \
       --config config/default_config.yml \
       --connections path/to/Connections.csv \
       --messages path/to/Messages.csv \
       --output linkedin_insights \
       --format all
   ```

4. **View the results**
   - Check generated visualizations in the `visualizations/` directory
   - Review the comprehensive report in your chosen format(s)

## 📊 Sample Output

```json
{
  "network_growth": {
    "total_connections": 500,
    "growth_trend": "Strong growth",
    "seasonal_patterns": {
      "highest_growth_month": "March",
      "lowest_growth_month": "December"
    }
  },
  "industry_distribution": {
    "Software Development": 25,
    "Data Science": 20,
    "Product Management": 15
  }
}
```

## 📈 Visualizations

The analyzer generates several types of visualizations:

1. Network Growth Chart
   ![Network Growth](docs/images/network_growth_sample.png)

2. Industry Distribution
   ![Industry Distribution](docs/images/industry_distribution_sample.png)

3. Message Activity Heatmap
   ![Message Activity](docs/images/message_activity_sample.png)

4. Interactive Network Graph
   ![Network Graph](docs/images/network_graph_sample.png)

## 🛠️ Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Format code
black src/

# Check code quality
flake8 src/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

1. **Installation Problems**
   ```bash
   # For matplotlib issues on macOS
   pip install matplotlib --upgrade
   ```

2. **Visualization Libraries**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-tk
   
   # macOS
   brew install python-tk
   ```

3. **Data Format Issues**
   - Ensure CSV files are in UTF-8 encoding
   - Verify column names match LinkedIn export format

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project
- Built with inspiration from the LinkedIn developer community
- Visualization libraries: matplotlib, seaborn, plotly, networkx

## 📮 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/linkedin-network-analyzer](https://github.com/yourusername/linkedin-network-analyzer)

## 🗺️ Roadmap

- [ ] Add machine learning-based connection recommendations
- [ ] Implement natural language processing for message analysis
- [ ] Add integration with LinkedIn API (when available)
- [ ] Create web interface for analysis results
- [ ] Add export to more formats (e.g., PowerPoint, Excel)

---
⭐️ If you find this project useful, please consider giving it a star on GitHub!
