import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
from datetime import datetime, timedelta
import warnings
import calendar
import os

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="StockSmart Analytics",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

def format_currency(amount):
    """Format currency in Indian format (Lakhs and Crores)"""
    if amount >= 10000000:  # 1 Crore
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.2f}"

def preprocess_mongodb_export(data):
    """Convert MongoDB export format to regular JSON"""
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                if '$date' in value:
                    data[key] = value['$date']
                elif '$numberLong' in value:
                    data[key] = int(value['$numberLong'])
                else:
                    preprocess_mongodb_export(value)
            elif isinstance(value, list):
                for item in value:
                    preprocess_mongodb_export(item)
    return data

@st.cache_data
def load_data():
    """Load and preprocess bills and inventory data"""
    try:
        # Debug setting
        debug = st.secrets.get('debug', False)
        
        # Load bills data
        bills_file = st.secrets.paths.bills_data if hasattr(st.secrets, 'paths') else 'stocksmart_bills.json'
        
        with open(bills_file, 'r') as f:
            bills_data = json.load(f)
            bills_data = [preprocess_mongodb_export(bill.copy()) for bill in bills_data]
        
        # Convert bills data to DataFrame
        df_bills = pd.json_normalize(bills_data)
        
        # Handle date
        date_columns = ['date.$date', 'date.date', 'date']
        date_col = next((col for col in date_columns if col in df_bills.columns), None)
        if date_col:
            df_bills['date'] = pd.to_datetime(df_bills[date_col])
        
        # Process products
        df_bills = df_bills.explode('products')
        df_bills['product_name'] = df_bills['products'].apply(lambda x: x.get('product', 'Unknown') if isinstance(x, dict) else 'Unknown')
        df_bills['quantity'] = df_bills['products'].apply(lambda x: float(x.get('quantity', 0)) if isinstance(x, dict) else 0)
        df_bills['price'] = df_bills['products'].apply(lambda x: float(x.get('price', 0)) if isinstance(x, dict) else 0)
        df_bills['product_profit'] = df_bills['products'].apply(lambda x: float(x.get('profit', 0)) if isinstance(x, dict) else 0)
        df_bills['total_amount'] = df_bills['quantity'] * df_bills['price']
        
        # Load inventory data
        inventory_file = st.secrets.paths.inventory_data if hasattr(st.secrets, 'paths') else 'stocksmart_inventory.csv'
        df_inventory = pd.read_csv(inventory_file)
        
        return df_bills, df_inventory
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        if debug:
            st.write(f"Error details: {traceback.format_exc()}")
        return None, None

def get_next_month_dates():
    """Get the start and end dates for next month"""
    today = datetime.now()
    first_of_next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    last_day = calendar.monthrange(first_of_next_month.year, first_of_next_month.month)[1]
    last_of_next_month = first_of_next_month.replace(day=last_day)
    return first_of_next_month, last_of_next_month

def predict_stock(df_bills):
    """Predict stock requirements for next month based on historical patterns"""
    if df_bills is None or df_bills.empty:
        st.error("No bill data available for predictions")
        return pd.DataFrame()
    
    predictions = []
    next_month_start, next_month_end = get_next_month_dates()
    next_month_name = next_month_start.strftime("%B %Y")
    
    for product in df_bills['product_name'].unique():
        try:
            # Get product data
            product_data = df_bills[df_bills['product_name'] == product]
            
            # Calculate monthly statistics
            monthly_stats = (product_data.groupby(pd.Grouper(key='date', freq='M'))
                           ['quantity'].agg(['sum', 'mean', 'std']))
            
            if len(monthly_stats) < 1:
                continue
                
            # Calculate trend
            monthly_stats['month_num'] = range(len(monthly_stats))
            trend = np.polyfit(monthly_stats['month_num'], monthly_stats['sum'], 1)[0]
            
            # Calculate seasonality factor (if enough data)
            current_month = datetime.now().month
            next_month = (current_month % 12) + 1
            
            seasonal_factor = 1.0
            if len(monthly_stats) >= 12:
                monthly_factors = (product_data.groupby(product_data['date'].dt.month)
                                 ['quantity'].mean() / product_data['quantity'].mean())
                seasonal_factor = monthly_factors.get(next_month, 1.0)
            
            # Calculate base prediction
            base_prediction = monthly_stats['sum'].mean()
            
            # Apply trend and seasonality
            trend_adjustment = trend * len(monthly_stats)
            predicted_demand = (base_prediction + trend_adjustment) * seasonal_factor
            
            # Add safety stock based on variability
            safety_stock = monthly_stats['std'].mean() * 1.645  # 95% service level
            
            # Calculate recommended stock
            recommended_stock = int(np.ceil(max(1, predicted_demand + safety_stock)))
            
            # Calculate confidence metrics
            forecast_std = monthly_stats['std'].mean()
            confidence_lower = max(0, int(predicted_demand - 1.96 * forecast_std))
            confidence_upper = int(predicted_demand + 1.96 * forecast_std)
            
            # Get recent price for revenue projection
            recent_price = product_data['price'].iloc[-1] if not product_data.empty else 0
            projected_revenue = recommended_stock * recent_price
            
            predictions.append({
                'product_name': product,
                'recommended_stock': recommended_stock,
                'predicted_demand': int(predicted_demand),
                'confidence_range': f"{confidence_lower}-{confidence_upper}",
                'projected_revenue': projected_revenue,
                'trend': 'Increasing' if trend > 0 else 'Decreasing' if trend < 0 else 'Stable'
            })
            
        except Exception as e:
            continue
    
    return pd.DataFrame(predictions)

def get_top_products(df_bills, n=3):
    """Calculate top selling products with total revenue"""
    return (df_bills.groupby('product_name')
            .agg({
                'quantity': 'sum',
                'total_amount': 'sum',
                'product_profit': 'sum'
            })
            .sort_values('quantity', ascending=False)
            .head(n))

def categorize_demand(df_bills):
    """Categorize products by demand with detailed metrics"""
    product_metrics = df_bills.groupby('product_name').agg({
        'quantity': ['sum', 'mean', 'std'],
        'total_amount': 'sum',
        'product_profit': 'sum'
    }).round(2)
    
    product_metrics.columns = ['total_quantity', 'avg_quantity', 'std_quantity', 'total_revenue', 'total_profit']
    
    # Remove duplicate entries by selecting first occurrence
    product_metrics = product_metrics.loc[~product_metrics.index.duplicated(keep='first')]
    
    def get_category(row):
        if row['total_quantity'] < product_metrics['total_quantity'].quantile(0.33):
            return "Low"
        elif row['total_quantity'] < product_metrics['total_quantity'].quantile(0.67):
            return "Medium"
        else:
            return "High"
    
    product_metrics['demand_category'] = product_metrics.apply(get_category, axis=1)
    return product_metrics

def main():
    st.markdown('<p class="big-font">StockSmart Analytics Dashboard</p>', unsafe_allow_html=True)
    
    # Load data
    df_bills, df_inventory = load_data()
    if df_bills is None or df_inventory is None:
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Sales Analysis", "📊 Demand Analysis", "🔮 Stock Predictions"])
    
    with tab1:
        st.header("Sales Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Top 3 Selling Products")
            top_products = get_top_products(df_bills)
            fig_top = px.bar(
                top_products,
                y=['quantity', 'product_profit'],
                title="Top Selling Products by Quantity and Profit",
                barmode='group',
                color_discrete_sequence=['#2E86C1', '#28B463']
            )
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.subheader("Key Metrics")
            total_sales = df_bills['total_amount'].sum()
            total_profit = df_bills['product_profit'].sum()
            total_products = df_bills['quantity'].sum()
            avg_transaction = df_bills.groupby('_id')['total_amount'].sum().mean()
            
            st.metric("Total Sales", format_currency(total_sales))
            st.metric("Total Profit", format_currency(total_profit))
            st.metric("Total Products Sold", f"{total_products:,}")
            st.metric("Avg Transaction", format_currency(avg_transaction))
        
        # Sales Trends
        st.subheader("Sales Trends")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Daily sales trend
            daily_sales = df_bills.groupby('date')['total_amount'].sum().reset_index()
            fig_daily = px.line(
                daily_sales,
                x='date',
                y='total_amount',
                title="Daily Sales Trend",
                line_shape='spline'
            )
            fig_daily.update_layout(yaxis_title="Amount (₹)")
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col4:
            # Monthly sales trend
            monthly_sales = (df_bills.groupby(pd.Grouper(key='date', freq='M'))
                           ['total_amount'].sum().reset_index())
            fig_monthly = px.line(
                monthly_sales,
                x='date',
                y='total_amount',
                title="Monthly Sales Trend",
                line_shape='spline'
            )
            fig_monthly.update_layout(yaxis_title="Amount (₹)")
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab2:
        st.header("Demand Analysis")
        
        # Product Demand Categories
        demand_metrics = categorize_demand(df_bills)
        
        # Pie chart of demand categories
        demand_counts = demand_metrics['demand_category'].value_counts()
        fig_pie = px.pie(
            values=demand_counts.values,
            names=demand_counts.index,
            title="Product Demand Distribution",
            hole=0.4
        )
        st.plotly_chart(fig_pie)
        
        # Detailed demand metrics
        st.subheader("Detailed Demand Metrics")
        
        # Format currency columns
        display_metrics = demand_metrics.copy()
        display_metrics['total_revenue'] = display_metrics['total_revenue'].apply(format_currency)
        display_metrics['total_profit'] = display_metrics['total_profit'].apply(format_currency)
        display_metrics.index.name = 'Product'
        st.dataframe(display_metrics, use_container_width=True)
    
    with tab3:
        next_month_start, next_month_end = get_next_month_dates()
        st.header(f"Stock Predictions for {next_month_start.strftime('%B %Y')}")
        
        # Get predictions
        predictions_df = predict_stock(df_bills)
        
        if not predictions_df.empty:
            # Format currency columns
            predictions_df['projected_revenue'] = predictions_df['projected_revenue'].apply(
                lambda x: f"₹{x:,.2f}" if x > 0 else "N/A"
            )
            
            # Display predictions
            display_df = predictions_df[[
                'product_name', 
                'recommended_stock', 
                'predicted_demand', 
                'confidence_range',
                'trend',
                'projected_revenue'
            ]]
            
            display_df.columns = [
                'Product Name',
                'Recommended Stock',
                'Predicted Demand',
                'Confidence Range',
                'Trend',
                'Projected Revenue'
            ]
            
            st.dataframe(display_df, use_container_width=True)
            
            # Summary statistics
            total_recommended = display_df['Recommended Stock'].sum()
            st.subheader("Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Stock Needed", f"{total_recommended:,}")
            with col2:
                increasing_products = (predictions_df['trend'] == 'Increasing').sum()
                st.metric("Products with Increasing Demand", increasing_products)
            with col3:
                st.metric("Forecast Period", next_month_start.strftime('%B %Y'))
            
        else:
            st.warning("Not enough data to make predictions")

if __name__ == "__main__":
    main()