import csv
import os
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

# Load customer data from CSV
def load_customers_from_csv():
    customers = []
    product_columns = [
        'casa', 'fd', 'vay_mua_oto', 'vay_tieu_dung', 'vay_sxkd',
        'vay_mua_bds', 'vay_dac_thu', 'vay_khac',
        'amt_credit_card', 'amt_debit_atm_card', 'amt_debit_post_card'
    ]
    with open(os.path.join(os.path.dirname(__file__), '../customer_recommendations_output.csv'), mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            customer = {
                "id": row["customer_id"],
                "age": int(row["age"]),
                "income": float(row["thu_nhap"]),
                "segment": row["customer_segment"],
                "recommended_product": row["best_next_product"],  
                "purchase_history": {}
            }
            # Add product amounts to the purchase history
            for product, amount in row.items():
                if product in product_columns:
                    customer["purchase_history"][product] = float(amount)
            customers.append(customer)

    return customers

# Load customers once at startup
customers = load_customers_from_csv()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get-customers", methods=["GET"])
def get_customers():
    return jsonify(customers)

@app.route("/get-customer/<customer_id>", methods=["GET"])
def get_customer(customer_id):
    customer = next((c for c in customers if c["id"] == customer_id), None)
    if not customer:
        return jsonify({"error": "Customer not found"}), 404
    return jsonify(customer)

if __name__ == "__main__":
    app.run(debug=True)
