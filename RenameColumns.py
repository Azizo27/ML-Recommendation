
import pandas as pd
import numpy as np



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
