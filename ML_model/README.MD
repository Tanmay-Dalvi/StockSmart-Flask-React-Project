# StockSmart Analytics 📊

A machine learning-powered inventory and sales analytics dashboard built with Streamlit. StockSmart helps businesses analyze sales patterns, track inventory, and make data-driven stocking decisions.

## 🌟 Features

- 📈 Real-time sales analysis and visualization
- 📊 Demand analysis with product categorization
- 🔮 ML-powered stock predictions for the next month
- 💹 Revenue and profit tracking
- 📉 Interactive charts and graphs

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stocksmart-analytics.git
cd stocksmart-analytics
```

2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

4. Set up your data files
- Place your bills data in `stocksmart_bills.json`
- Place your inventory data in `stocksmart_inventory.csv`

5. Create a `secrets.toml` file in the `.streamlit` folder with your configuration:
```toml
[paths]
bills_data = "stocksmart_bills.json"
inventory_data = "stocksmart_inventory.csv"

[app]
debug = false
```

### Running the Application

1. Start the Streamlit server
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure

```
stocksmart-analytics/
├── app.py                     # Main application file
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── secrets.toml          # Configuration file
├── stocksmart_bills.json     # Sample bills data
├── stocksmart_inventory.csv  # Sample inventory data
└── README.md                 # Project documentation
```

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - The web framework used
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation
- [scikit-learn](https://scikit-learn.org/) - Machine learning tools
- [statsmodels](https://www.statsmodels.org/) - Statistical models

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
