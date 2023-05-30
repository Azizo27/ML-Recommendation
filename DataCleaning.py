import pandas as pd
import numpy as np
import random
from SegmentationPrediction import predict_segmentation
import os

def Init_csv(file_name, small_file_name,  nrows):
    # Count the total number of rows in the large .csv file
    total_rows = sum(1 for _ in open(file_name)) - 1  # subtract 1 to exclude the header row
    sample_indices = random.sample(range(1, total_rows + 1), nrows)  # add 1 to include the header row

    print("Writing the header row to the small .csv file...")
    # Write the header row to the small .csv file
    with open(small_file_name, 'w') as f:
        with open(file_name) as f_large:
            header = f_large.readline().replace('"', '')
            f.write(header)

    print("Writing the randomly selected rows to the small .csv file...")
    # Read and append only the randomly sampled rows to the small .csv file
    with open(small_file_name, 'a') as f:
        with open(file_name) as f_large:
            for i, line in enumerate(f_large):
                if i + 1 in sample_indices:
                    f.write(line)


#This function loads the data ENTIRELY from a csv. Also, It can be filtered by date May/June. Also, It can copy to a new csv file by renaming the columns
def LoadCsv(file_name, new_file_name):
    chunksize = 100000
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, low_memory=False, chunksize=chunksize):
        '''
        #TO FILTER BY DATE (NB: YOU MUST DELETE THE NEXT LINE)
        filtered_chunk = chunk[chunk['fecha_dato'].astype(str).str.startswith(('2015-06'))]
        chunks.append(filtered_chunk)
        '''
        chunks.append(chunk)

    print("Data loaded. Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    if not os.path.exists(new_file_name):
        print("Renaming columns...")
        df = RenameColumns(df)
        print("Cleaning data...")
        df = CleaningData(df)
        print("Saving to a new csv file...")
        df.to_csv(new_file_name, index=False)
        
    return df

#This function renames the columns of the data
def RenameColumns(df):
    column_names = {
        'fecha_dato': 'date',
        'ncodpers': 'customer_code',
        'ind_empleado': 'employee_index',
        'pais_residencia': 'country_residence',
        'sexo': 'gender',
        'age': 'age',
        'fecha_alta': 'customer_start_date',
        'ind_nuevo': 'new_customer_index',
        'antiguedad': 'customer_seniority',
        'indrel': 'primary_customer_index',
        'ult_fec_cli_1t': 'last_date_as_primary_customer',
        'indrel_1mes': 'customer_type_at_beginning_of_month',
        'tiprel_1mes': 'customer_relation_type_at_beginning_of_month',
        'indresi': 'residence_index',
        'indext': 'foreigner_index',
        'conyuemp': 'spouse_index',
        'canal_entrada': 'channel_used_by_customer_to_join',
        'indfall': 'deceased_index',
        'tipodom': 'address_type',
        'cod_prov': 'province_code',
        'nomprov': 'province_name',
        'ind_actividad_cliente': 'activity_index',
        'renta': 'gross_income',
        'segmento': 'segmentation',
        'ind_ahor_fin_ult1': 'product_savings_account',
        'ind_aval_fin_ult1': 'product_guarantees',
        'ind_cco_fin_ult1': 'product_current_accounts',
        'ind_cder_fin_ult1': 'product_derivada_account',
        'ind_cno_fin_ult1': 'product_payroll_account',
        'ind_ctju_fin_ult1': 'product_junior_account',
        'ind_ctma_fin_ult1': 'product_mas_particular_account',
        'ind_ctop_fin_ult1': 'product_particular_account',
        'ind_ctpp_fin_ult1': 'product_particular_plus_account',
        'ind_deco_fin_ult1': 'product_short_term_deposits',
        'ind_deme_fin_ult1': 'product_medium_term_deposits',
        'ind_dela_fin_ult1': 'product_long_term_deposits',
        'ind_ecue_fin_ult1': 'product_e_account',
        'ind_fond_fin_ult1': 'product_funds',
        'ind_hip_fin_ult1': 'product_mortgage',
        'ind_plan_fin_ult1': 'product_first_pensions',
        'ind_pres_fin_ult1': 'product_loans',
        'ind_reca_fin_ult1': 'product_taxes',
        'ind_tjcr_fin_ult1': 'product_credit_card',
        'ind_valo_fin_ult1': 'product_securities',
        'ind_viv_fin_ult1': 'product_home_account',
        'ind_nomina_ult1': 'product_payroll',
        'ind_nom_pens_ult1': 'product_second_pensions',
        'ind_recibo_ult1': 'product_direct_debit'
    }
    df.rename(columns=column_names, inplace=True)
    return df

#This function displays the information of the data
def DisplayInformation(df):
    print("Info:", df.info())
    # Display unique values for each column
    for col in df.columns:
        unique_values = df[col].unique()
        print("Unique values in the '{}' column:".format(col))
        null_count = df[col].isnull().sum()
        print("Number of null values in the column ",col," :  ",null_count)
        print(unique_values)


#This function cleans the data.
#NB: I added "U", "UKN" or "UKNOWN" for Unknown to fill some null columns
def CleaningData(df):
    
    print("Cleaning Date...")
    df["date"] = pd.to_datetime(df["date"],format="%Y-%m-%d")
    
    print("Nothing To Clean In Customer Code...")
    
    print("Cleaning Employee Index...")
    #NB: In the original instruction, Employee Index should have five types: A, B, F, N, P. 
    #    However, this column only have does not have P values but it does have S
    df.dropna(subset=['employee_index'], inplace=True)
    
    print("Nothing To Clean In Country Residence...")
    
    print("Cleaning Gender...")
    df['gender'].fillna('U', inplace=True)
    
    
    print("Cleaning Age...")
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df["age"] = np.where(df["age"] > 100, 100, df["age"])
    df["age"] = np.where(df["age"] < 0, 0, df["age"])
    
    print("Cleaning Customer Start Date...")
    # NB: There is no Null values in this column But I fill it in case of
    df["customer_start_date"].fillna(pd.NaT, inplace=True)
    df["customer_start_date"] = pd.to_datetime(df["customer_start_date"],format="%Y-%m-%d")
    
    print("Cleaning New Customer Index...")
    #NB: At this step, There is no null values in this column 
    # but I  do a fillna just in case.
    df['new_customer_index'].fillna(0, inplace=True)
    df['new_customer_index'] = df['new_customer_index'].astype(int)
    
    print("Cleaning Customer Seniority...")
    df['customer_seniority'] = pd.to_numeric(df['customer_seniority'], errors='coerce')
    df["customer_seniority"] = np.where(df["customer_seniority"] < 0, 0, df["customer_seniority"])
    df["customer_seniority"] = np.where(df["customer_seniority"] > 1000, 1000, df["customer_seniority"])
    df['customer_seniority'].fillna(0, inplace=True)
    
    print("Cleaning Primary Customer Index...")
    # NB: Even If there is no null values, How to fill the null values in this column? (probably put it at 0)
    df['primary_customer_index'] = df['primary_customer_index'].astype(int)
    
    
    print("Cleaning Last Date as Primary Customer (Deleting the column) ...")
    '''
    # NB: We will not use this column in our model
    df["last_date_as_primary_customer"].fillna(pd.NaT, inplace=True)
    df["last_date_as_primary_customer"] = pd.to_datetime(df["last_date_as_primary_customer"],format="%Y-%m-%d")
    '''
    # I will drop this column because it is useless for our model
    df = df.drop('last_date_as_primary_customer', axis=1)
    
    
    print("Cleaning Customer Type at Beginning of Month...")
    df['customer_type_at_beginning_of_month'] = df['customer_type_at_beginning_of_month'].replace({'1.0': '1', 1.0: '1', 1: '1',
                                                                                               '2.0': '2', 2.0: '2', 2: '2',
                                                                                               '3.0': '3', 3.0: '3', 3: '3',
                                                                                               '4.0': '4', 4.0: '4', 4: '4'})
    df['customer_type_at_beginning_of_month'].fillna('U', inplace=True)
    
    print("Cleaning Customer Relation Type at Beginning of Month...")
    #NB: There is a value "N" in this column, which is not in the original instruction
    df['customer_relation_type_at_beginning_of_month'].fillna('U', inplace=True)
    
    print("Cleaning Residence Index...")
    
    print("Cleaning Foreigner Index...")
    
    print("Cleaning Spouse Index...")
    #NB: In the original instruction, Spouse Index should be equal to 1 if the customer is spouse of an employee. 
    #    However, this column only have N/S values, for No and Si (Yes in Spanish).
    df['spouse_index'].fillna('N', inplace=True)
    
    print("Cleaning Channel Used by Customer to Join...")
    df['channel_used_by_customer_to_join'].fillna('UNK', inplace=True)
    
    print("Cleaning Deceased Index...")
    
    print("Cleaning Address Type...")
    df['address_type'].fillna(0, inplace=True)
    df['address_type'] = df['address_type'].astype(int)
    
    print("Cleaning Province Code...")
    df['province_code'].fillna(99, inplace=True)
    df['province_code'] = df['province_code'].astype(int)
    
    print("Cleaning Province Name...")
    df['province_name'] = df['province_name'].str.replace(',', '')
    df['province_name'].fillna('UNKNOWN', inplace=True)
    
    print("Cleaning Activity Index...")
    #NB: At this step, There is no null values in this column. 
    # All of it is equal to 1 but I do a fillna just in case
    df['activity_index'].fillna(0, inplace=True)
    df['activity_index'] = df['activity_index'].astype(int)
    
    print("Cleaning Segmentation...")
    df = predict_segmentation(df)
    
    print("Cleaning Gross Income...")
    #At the end, I decided to drop all the rows with null values in this column
    df['gross_income'] = df['gross_income'].str.replace(' ', '')
    df['gross_income'] = pd.to_numeric(df['gross_income'], errors='coerce')
    df.dropna(subset=['gross_income'], inplace=True)

    products_name = [col for col in df.columns if col.startswith('product_')]
    
    if len(products_name) != 0: # If there is products columns
        print("Cleaning All Products...")
        df.loc[:, products_name] = df.loc[:, products_name].fillna(0)
        df['product_payroll'] = df['product_payroll'].astype(int)
        df['product_second_pensions'] = df['product_second_pensions'].astype(int)
    
    print("Data Cleaning Done !")
    
    return df

