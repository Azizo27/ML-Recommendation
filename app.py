import pandas as pd
from flask import Flask, render_template, request
from PredictProbabilityProduct import PredictProbabilityProduct
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    
    month = request.form['month']
    age = int(request.form['age'])
    gross_income = float(request.form['gross_income'])
    customer_seniority = int(request.form['customer_seniority'])
    customer_relation_type= request.form['customer_relation_type_at_beginning_of_month']
    segmentation = request.form['segmentation']
    gender = request.form['gender']
    
    
    
    # Create a dataframe with the sent elements
    data = { 'age': [age],
            'gross_income': [gross_income], 
            'customer_seniority': [customer_seniority], 
            'customer_relation_type_at_beginning_of_month': [customer_relation_type], 
            'segmentation':[segmentation],  
            'gender': [gender]}
    df = pd.DataFrame(data)

    # Process the selected option and dataframe here (you can replace this with your own logic)
    
    all_products=  [ "product_savings_account", "product_guarantees", "product_current_accounts"]
    
    
    for target in all_products:
        target_column = PredictProbabilityProduct(df, target, month)
        df[target] = target_column
        print("Done with", target)
    
    #result = f'Selected option: {selected_option}<br>'
    
    result = 'Dataframe:<br>'
    result += df.to_html()

    return result

if __name__ == '__main__':
    app.run(debug=True)
