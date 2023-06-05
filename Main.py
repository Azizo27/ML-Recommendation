from LoadCsv import LoadCsv
from PredictProbabilityProduct import PredictProbabilityProduct
from MergePredictions import MergePredictions
from RankRecommendation import RankRecommendation
from EncodingFeatures import EncodingAllFeatures
import pandas as pd
import numpy as np

'''
if __name__ == '__main__':
    dfTopredict = LoadCsv("Cleaned_Renamed_test_May2016.csv", "Cleaned_Renamed_test_May2016.csv")

    features = ['age', 'gross_income', 'customer_seniority', 'customer_relation_type_at_beginning_of_month', 'segmentation', 'gender']
    FileToCreateModel = "Cleaned_Renamed_train_May2015.csv"

    all_products=  [ "product_savings_account", "product_guarantees", "product_current_accounts",
        "product_derivada_account", "product_payroll_account", "product_junior_account",
        "product_mas_particular_account", "product_particular_account", "product_particular_plus_account",
        "product_short_term_deposits", "product_medium_term_deposits", "product_long_term_deposits",
        "product_e_account", "product_funds", "product_mortgage", "product_first_pensions",
        "product_loans", "product_taxes", "product_credit_card", "product_securities",
        "product_home_account", "product_payroll", "product_second_pensions", "product_direct_debit"]
    
    for target in all_products:
        model_file_name = 'model_' + target + '.pkl'
        PredictProbabilityProduct(dfTopredict, features, target, model_file_name, FileToCreateModel)
        print("Done with", target)
    
    
    print("Merging predictions...")
    MergedDf = MergePredictions()
    print("Ranking recommendations...")
    RankRecommendation(MergedDf)
'''