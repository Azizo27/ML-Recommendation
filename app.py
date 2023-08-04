import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, jsonify
from PredictProbabilityProduct import PredictProbabilityProduct
from LoadCsv import LoadCsv
import threading
import os
import time
import signal
from prediction.calculations import exponential_smoothing_with_date


app = Flask(__name__, static_folder='static')

result = None
df = None
calculation_complete = threading.Event()


all_products=  [ "product_savings_account", "product_guarantees", "product_current_accounts",
        "product_derivada_account", "product_payroll_account", "product_junior_account",
        "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
        "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
        "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
        "product_loans", "product_taxes", "product_credit_card", "product_securities",
        "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    message = ""
    final_df = None
    valid_customer = False
        
    if request.method == 'POST':
        customer_id = int(request.form['customer_id'])
        
        global df
        if df is None:
            df = LoadCsv("Cleaned_Renamed_train_ver2.csv", "Cleaned_Renamed_train_ver2.csv")
            
        # Convert 'customer_code' column to int (if it's not already)
        if df['customer_code'].dtype != int:
            df['customer_code'] = df['customer_code'].astype(int)

        if customer_id in df['customer_code'].unique():
            final_df = df[df['customer_code'] == customer_id].copy()
            
            print("Doing sorting...")
            final_df = final_df.sort_values(by=['customer_code', 'date'])


            alpha = 0.7  # You can adjust this value based on your preference

            print("Applying exponential smoothing...")
            # Apply exponential smoothing to the entire DataFrame
            
            smoothed_values = {}
            for col in all_products:
                smoothed_values[col] = exponential_smoothing_with_date(final_df[col].values, alpha)[-1]
            final_df= pd.DataFrame({'customer_code': df['customer_code'].iloc[0], **smoothed_values}, index=[0])

            print("Dropping duplicates...")
            # Drop duplicates based on customer_code
            final_df = final_df.drop_duplicates(subset='customer_code')
            
            final_df.drop(columns=['customer_code'], inplace=True)
            valid_customer = True
        else:
            message = "Customer ID does not exist in the DataFrame."

    return render_template('prediction.html', message=message, valid_customer=valid_customer, result=final_df)


@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')


@app.route('/calculate_recommendation', methods=['POST'])
def calculate_recommendation(month, age, gross_income, customer_seniority, customer_relation_type, segmentation, gender):
    global result

    # Create a dataframe with the sent elements
    data = {
        'age': [age],
        'gross_income': [gross_income],
        'customer_seniority': [customer_seniority],
        'customer_relation_type_at_beginning_of_month': [customer_relation_type],
        'segmentation': [segmentation],
        'gender': [gender]
    }
    df = pd.DataFrame(data)

    for target in all_products:
        target_column = PredictProbabilityProduct(df, target, month)
        df[target] = target_column
        print("Done with", target)

    columns_to_keep = df.columns[df.columns.str.startswith('product_')]
    df = df[columns_to_keep]
    
    highest_values = df.iloc[0].sort_values(ascending=False)
    df_ordered = df[highest_values.index]
    
    result = df_ordered.to_html()
    calculation_complete.set()  # Set the event to indicate that the calculation is complete



@app.route('/result')
def show_result():
    global result
    if calculation_complete.is_set() and result is not None:
        result = pd.read_html(result)[0] # Retransform the html table into a dataframe
        result = result.drop("Unnamed: 0", axis=1)  # Drop the column that is created by default during the conversion
        return render_template('result.html', result=result)
    else:
        return render_template('waiting.html')


@app.route('/check_result')
def check_result():
    global result

    if calculation_complete.is_set() and result is not None:
        return jsonify({'result_available': True, 'result': result})
    else:
        return jsonify({'result_available': False})


# Modify the /process route to include the redirect to /result
@app.route('/process_recommendation', methods=['POST'])
def process_recommendation():
    global result

    month = request.form['month']
    age = int(request.form['age'])
    gross_income = float(request.form['gross_income'])
    customer_seniority = int(request.form['customer_seniority'])
    customer_relation_type = request.form['customer_relation_type_at_beginning_of_month']
    segmentation = request.form['segmentation']
    gender = request.form['gender']

    calculation_complete.clear()  # Clear the event before starting the new thread
    threading.Thread(target=calculate_recommendation, args=(month, age, gross_income, customer_seniority,
                                                        customer_relation_type, segmentation, gender)).start()

    return redirect(url_for('show_result'))  # Redirect to the show_result route


if __name__ == '__main__':
    app.run(debug=True)
