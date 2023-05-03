import pandas as pd
import numpy as np
import sqlalchemy

#This function loads the data ENTIRELY from a csv. Also, It can be filtered by date May/June. Also, It can copy to a new csv file by renaming the columns
def LoadCsv(file_name, new_file_name):
    chunksize = 100000
    chunks = []
    
    print("Loading data...")
    for chunk in pd.read_csv(file_name, low_memory=False, chunksize=chunksize):
        '''
        #TO FILTER BY DATE (NB: YOU MUST DELETE THE NEXT LINE)
        filtered_chunk = chunk[chunk['fecha_dato'].astype(str).str.startswith(('2015-05', '2015-06'))]
        chunks.append(filtered_chunk)
        '''
        chunks.append(chunk)

    print("Data loaded. Concatenating chunks...")
    df = pd.concat(chunks, ignore_index=True)
    
    '''
    #TO RENAME THE COLUMNS, CLEANS DATA AND SAVE TO A NEW CSV FILE
    print("Renaming columns...")
    df = RenameColumns(df)
    #print("Cleaning data...")
    #df = CleaningData(df)
    print("Saving to a new csv file...")
    df.to_csv(new_file_name, index=False)
    '''
    
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
#We send a chunk of data to this function, and it cleans it.
def CleaningData(df):
    
    print("Cleaning Employee Index...")
    df.dropna(subset=['employee_index'], inplace=True)
    
    print("Cleaning Country Residence...")
    
    print("Cleaning Gender...")
    
    print("Cleaning Age...")
    
    print("Cleaning Customer Start Date...")
    
    print("Cleaning New Customer Index...")
    
    print("Cleaning Customer Seniority...")
    df['customer_seniority'] = pd.to_numeric(df['customer_seniority'], errors='coerce')
    df["customer_seniority"] = np.where(df["customer_seniority"] < 0, 0, df["customer_seniority"])
    df['customer_seniority'].fillna(0, inplace=True)
    
    print("Cleaning Primary Customer Index...")
    
    print("Cleaning Last Date as Primary Customer...")
    
    print("Cleaning Customer Type at Beginning of Month...")
    
    print("Cleaning Customer Relation Type at Beginning of Month...")
    
    print("Cleaning Residence Index...")
    
    print("Cleaning Foreigner Index...")
    
    print("Cleaning Spouse Index...")
    
    print("Cleaning Channel Used by Customer to Join...")
    
    print("Cleaning Deceased Index...")
    
    print("Cleaning Address Type...")
    
    print("Cleaning Province Code...")
    
    print("Cleaning Province Name...")
    
    print("Cleaning Activity Index...")
    
    print("Cleaning Gross Income...")
    
    print("Cleaning Segmentation...")
    
    print("Cleaning All Products...")
    
    df["date"] = pd.to_datetime(df["date"],format="%Y-%m-%d")
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df["customer_start_date"] = pd.to_datetime(df["customer_start_date"],format="%Y-%m-%d")
    df["last_date_as_primary_customer"] = pd.to_datetime(df["last_date_as_primary_customer"],format="%Y-%m-%d")
    df['customer_type_at_beginning_of_month'] = df['customer_type_at_beginning_of_month'].replace({'1.0': '1', 1.0: '1',
                                                                                               '2.0': '2', 2.0: '2',
                                                                                               '3.0': '3', 3.0: '3',
                                                                                               '4.0': '4', 4.0: '4'})
    
    return df
    

print("Importing libraries...")
df = LoadCsv("CsvFiles/CountDataset.csv", "Renamed_train_ver2.csv")
df.dropna(subset=['employee_index'], inplace=True)
DisplayInformation(df)
