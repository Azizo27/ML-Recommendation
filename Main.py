from LoadCsv import LoadCsv
from PredictProbabilityProduct import PredictProbabilityProduct
from CreatingModelProduct import CreatingAllMonthModels
from Useless.MergePredictions import MergePredictions
from RankRecommendation import RankRecommendation
from DisplayInformation import DisplayInformation
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    
    all_products=  [ "product_savings_account", "product_guarantees", "product_current_accounts",
        "product_derivada_account", "product_payroll_account", "product_junior_account",
        "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
        "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
        "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
        "product_loans", "product_taxes", "product_credit_card", "product_securities",
        "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]
    
    all_months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    features = ['age', 'gross_income', 'customer_seniority', 'customer_relation_type_at_beginning_of_month', 'segmentation', 'gender']
    
    train_file_name = "Cleaned_Renamed_train_ver2.csv" 
    
    CreatingAllMonthModels(all_products, all_months, features, train_file_name)
    
    dfTopredict = LoadCsv("Cleaned_Renamed_test_May2016.csv", "Cleaned_Renamed_test_May2016.csv")
    
    selected_month = "May"
    merged_df = dfTopredict["customer_code"].copy()
    
    for target in all_products:
        target_column = PredictProbabilityProduct(dfTopredict, target, selected_month)
        merged_df = pd.merge(merged_df, target_column, left_index=True, right_index=True)
        print("Done with", target)
    
    print("Ranking recommendations...")
    RankRecommendation(merged_df)