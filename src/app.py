import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash
from flask_socketio import SocketIO
import pandas as pd
from datetime import date

app = Flask(__name__, static_folder='static', template_folder='templates')
app.secret_key = '123456'  # Change to a more secure key in production

# Initialize Flask-SocketIO
socketio = SocketIO(app)

# Load product data from all stores (for all_products.html)
def get_all_products():
    file_paths = [
        {'path': 'data/HM_All_Product_Sales.csv', 'store': 'H&M', 'category_path': 'data/raw/hm.csv'},
        {'path': 'data/ALLRetail_Store_1.csv', 'store': 'Retail Store 1', 'category_path': 'data/raw/store1.csv'},
        {'path': 'data/ALLRetail_Store_2.csv', 'store': 'Retail Store 2', 'category_path': 'data/raw/store2.csv'}
    ]
    all_products = []

    for file in file_paths:
        file_path = file['path']
        store_name = file['store']
        category_data_path = file['category_path']

        if os.path.exists(file_path):
            # Load the main product data
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            # Normalize ProductName for consistent matching
            data['ProductName'] = data['ProductName'].str.lower().str.strip()
            products = data[['ProductName']].drop_duplicates(subset='ProductName')
            products['Store'] = store_name

            # Load category data for this store if available
            category_data = None
            if os.path.exists(category_data_path):
                category_data = pd.read_csv(category_data_path, encoding='utf-8')
                # Normalize ProductName for consistent matching
                category_data['ProductName'] = category_data['ProductName'].str.lower().str.strip()
                category_data = category_data[['ProductName', 'Category']].drop_duplicates(subset='ProductName')

            # Merge with category data if available
            if category_data is not None:
                products = products.merge(category_data, on='ProductName', how='left')
                products['Category'] = products['Category'].fillna('Uncategorized')
            else:
                products['Category'] = 'Uncategorized'

            all_products.append(products)

    if not all_products:
        return []

    # Combine all products into a single DataFrame and deduplicate
    combined_products = pd.concat(all_products).drop_duplicates(subset='ProductName')
    return combined_products.to_dict('records')

# Load only H&M products from two datasets (for index.html)
def get_hm_products():
    all_data_path = 'data/HM_All_Product_Sales.csv'  # Changed from ALL_H&M.csv
    category_data_path = 'data/raw/hm.csv'  # Contains ProductName + Category

    # Load HM_All_Product_Sales.csv if it exists
    if os.path.exists(all_data_path):
        all_data = pd.read_csv(all_data_path, encoding='ISO-8859-1')
        # Normalize ProductName for consistent matching
        all_data['ProductName'] = all_data['ProductName'].str.lower().str.strip()
        products = all_data[['ProductName']].drop_duplicates(subset='ProductName')
    else:
        return []

    # Load H_M.csv if it exists
    category_data = None
    if os.path.exists(category_data_path):
        category_data = pd.read_csv(category_data_path, encoding='utf-8')
        # Normalize ProductName for consistent matching
        category_data['ProductName'] = category_data['ProductName'].str.lower().str.strip()
        category_data = category_data[['ProductName', 'Category']].drop_duplicates(subset='ProductName')

    # Merge category info into the HM_All_Product_Sales data
    if category_data is not None:
        merged = products.merge(category_data, on='ProductName', how='left')
    else:
        merged = products

    # Handle any missing categories
    merged['Category'] = merged['Category'].fillna('Uncategorized')
    merged['Store'] = 'H&M'

    # Ensure no duplicates remain after merging
    merged = merged.drop_duplicates(subset='ProductName')

    # Debugging: Check for duplicates
    if merged['ProductName'].duplicated().any():
        print("Warning: Duplicates found in H&M products after merging:")
        print(merged[merged['ProductName'].duplicated()])

    return merged.to_dict('records')

# Load trending products from three datasets
def get_trending_products(limit=5):
    file_paths = [
        {'path': 'data/HM_All_Product_Sales.csv', 'store': 'H&M'},  # Changed from ALL_H&M.csv
        {'path': 'data/ALLRetail_Store_1.csv', 'store': 'Retail Store 1'},
        {'path': 'data/ALLRetail_Store_2.csv', 'store': 'Retail Store 2'}
    ]
    all_data = []

    for file in file_paths:
        file_path = file['path']
        store_name = file['store']
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            if 'Date' not in data.columns:
                print(f"Error: 'Date' column not found in {file_path}")
                continue
            
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data[data['Date'] > '2023-01-01']
            trending = data.groupby(['ProductName'])['Sales'].sum().reset_index()
            trending = trending.sort_values(by='Sales', ascending=False).head(limit)
            trending['Store'] = store_name
            all_data.append(trending)

    if not all_data:
        print("No data found after filtering.")
        return pd.DataFrame()

    combined_data = pd.concat(all_data)
    combined_data = combined_data.groupby(['Store', 'ProductName']).agg({'Sales': 'sum'}).reset_index()
    trending = combined_data.sort_values(by='Sales', ascending=False).head(15)
    return trending

# Load lowest-selling products from H&M dataset
def get_lowest_selling_products(limit=5):
    file_path = 'data/HM_All_Product_Sales.csv'  # Changed from ALL_H&M.csv
    if os.path.exists(file_path):
        data = pd.read_csv(file_path, encoding='ISO-8859-1')
        if 'Date' not in data.columns or 'Sales' not in data.columns:
            print(f"Missing 'Date' or 'Sales' column in {file_path}")
            return pd.DataFrame()

        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data[data['Date'] > '2023-01-01']
        lowest_sales = data.groupby('ProductName')['Sales'].sum().reset_index()
        lowest_sales = lowest_sales.sort_values(by='Sales', ascending=True).head(limit)
        lowest_sales['Store'] = 'H&M'
        return lowest_sales
    return pd.DataFrame()

@app.route('/')
@app.route('/index.html')
def index():
    products = get_hm_products()  # Get only H&M products
    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('index.html', products=products, today=today, dark_mode=dark_mode)

@app.route('/trending.html')
def trending():
    trending_products = get_trending_products(limit=5)
    if trending_products.empty:
        flash("No trending products found after 2023.")
        return redirect(url_for('index'))

    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('trending.html', products=trending_products.to_dict('records'), today=today, dark_mode=dark_mode)

@app.route('/updateAll', methods=['POST'])
def update_all():
    data = request.get_json()
    with open('update_data.json', 'w') as f:
        json.dump(data, f)
    socketio.emit('data_updated', data)
    return jsonify({'status': 'success'})

@app.route('/toggle-dark-mode', methods=['POST'])
def toggle_dark_mode():
    session['dark_mode'] = not session.get('dark_mode', False)
    return ('', 204)

@app.route('/recommend.html')
def recommend():
    file_paths = [
        {'path': 'data/ALLRetail_Store_1.csv', 'store': 'Retail Store 1'},
        {'path': 'data/ALLRetail_Store_2.csv', 'store': 'Retail Store 2'}
    ]
    top_sales_products = []

    for file in file_paths:
        file_path = file['path']
        store_name = file['store']
        if os.path.exists(file_path):
            data = pd.read_csv(file_path, encoding='ISO-8859-1')
            if 'Date' not in data.columns or 'Sales' not in data.columns:
                print(f"Missing 'Date' or 'Sales' column in {file_path}")
                continue

            data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
            data = data[data['Date'] > '2023-01-01']
            store_top_sales = data.groupby('ProductName')['Sales'].sum().reset_index()
            store_top_sales = store_top_sales.sort_values(by='Sales', ascending=False).head(5)
            store_top_sales['Store'] = store_name
            top_sales_products.append(store_top_sales)

    if not top_sales_products:
        flash("No data found for top-selling products in the stores.")
        return redirect(url_for('index'))

    top_sales_df = pd.concat(top_sales_products)

    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('recommend.html', top_sales=top_sales_df.to_dict('records'), today=today, dark_mode=dark_mode)

@app.route('/lowest.html')
def lowest():
    lowest_products = get_lowest_selling_products(limit=5)
    if lowest_products.empty:
        flash("No data found for lowest-selling products in H&M.")
        return redirect(url_for('index'))

    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('lowest.html', lowest_sales=lowest_products.to_dict('records'), today=today, dark_mode=dark_mode)

@app.route('/help.html')
def help():
    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('help.html', today=today, dark_mode=dark_mode)

@app.route('/all_products.html')
def all_products():
    products = get_all_products()  # Get products from all stores
    today = date.today().strftime("%B %d, %Y")
    dark_mode = 'dark' if session.get('dark_mode') else ''
    return render_template('all_products.html', products=products, today=today, dark_mode=dark_mode)

@app.route('/product_image/<product_name>')
def product_image(product_name):
    filename = product_name.replace(" ", "_") + ".png"
    image_path = os.path.join("static", "images", filename)
    if os.path.exists(image_path):
        return jsonify({"image_url": url_for('static', filename=f"images/{filename}")})
    return jsonify({"error": "Image not found"}), 404

@app.route('/product_sales/<product_name>')
def product_sales(product_name):
    from urllib.parse import unquote
    product_name = unquote(product_name)
    file_paths = ['data/HM_All_Product_Sales.csv', 'data/ALLRetail_Store_1.csv', 'data/ALLRetail_Store_2.csv']  # Changed from ALL_H&M.csv
    sales_data = []

    for file_path in file_paths:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, encoding='ISO-8859-1')
            if {'ProductName', 'Date', 'Sales'}.issubset(df.columns):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df[df['ProductName'] == product_name]
                df = df.dropna(subset=['Date'])
                if not df.empty:
                    grouped = df.groupby('Date')['Sales'].sum().reset_index()
                    sales_data.append(grouped)

    if not sales_data:
        return jsonify({"error": f"No sales data found for {product_name}"}), 404

    combined = pd.concat(sales_data).groupby('Date')['Sales'].sum().reset_index()
    combined = combined.sort_values(by='Date')
    return jsonify({
        "dates": combined['Date'].dt.strftime('%Y-%m-%d').tolist(),
        "quantities": combined['Sales'].tolist()
    })

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)