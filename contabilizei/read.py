import pandas as pd
import numpy as np
from pathlib import Path

# 1. Preamble.

data_path = Path(__file__).resolve().parent.parent / 'data' / \
    'Dados Case - Cientista de Dados [Contabilizei].xlsx'

# 2. Define continous and discrete variables.

variables_continuous = [
    'age',
    'n_access_simulator',
    'n_partners',
    'monthly_income',
    'tickets_opened',
    'tenure',
    'csat'
]

variables_discrete = [
    'propension',
    'gender_Feminino',
    'region_Centro-Oeste',
    'region_Nordeste',
    'region_Norte',
    'region_Sul',
    'customer_service_channel_Chat',
    'customer_service_channel_Email'
]

# 3. Function: import_data.

def import_data(
        dropna: bool = True,
        get_dummies: bool = True,
        drop_negative_monthly_income: bool = True,
        drop_dummy_gender = True,
        drop_dummy_region = True,
        drop_dummy_customer_service_channel = True
    ):
    """
    Import data on individual propension to make a deal.

    Args:
        dropna (bool): If True, all rows containing at least one column whose
            value is nan is dropped. Defaults to True.
        get_dummies (bool): If True, dummy variables are created. Defaults to
            True.
        drop_negative_monthly_income (bool, optional): If True, rows containing
            negative monthly incomes are dropped. Defaults to True.
        drop_dummy_gender: If True, the column 'gender_Masculino' is dropped.
            For fairness, this column has been randomly chosen. Defaults to
            True.
        drop_dummy_region: If True, the column 'region_Sudeste' is dropped.
            Defaults to True.
        drop_dummy_customer_service_channel: If True, the column
            'customer_service_channel_Telefone' is dropped. Defaults to True.
    """

    data = pd.read_excel(
        io=data_path,
        sheet_name='Página1',
        names=[
            'propension',
            'age',
            'gender',
            'region',
            'n_access_simulator',
            'n_partners',
            'monthly_income',
            'tickets_opened',
            'customer_service_channel',
            'tenure',
            'csat'
        ],
        usecols='A:K',
        dtype={
            'propension': pd.UInt8Dtype(),
            'age': pd.UInt8Dtype(),
            'gender': str,
            'region': str,
            'n_access_simulator': pd.UInt8Dtype(),
            'n_partners': pd.UInt8Dtype(),
            'monthly_income': np.float32,
            'tickets_opened': pd.UInt8Dtype(),
            'customer_service_channel': str,
            'tenure': pd.UInt8Dtype(),
            'csat': pd.UInt8Dtype()
        }
    )

    if dropna:
        data.dropna(inplace=True)

    if get_dummies:
        data = pd.get_dummies(
            data=data,
            dummy_na=False,
            columns=[
                'gender',
                'region',
                'customer_service_channel'
            ],
            dtype=pd.UInt8Dtype()
        )

        if drop_dummy_gender:
            data.drop(
                labels='gender_Masculino',
                axis=1,
                inplace=True
            )

        if drop_dummy_region:
            data.drop(
                labels='region_Sudeste',
                axis=1,
                inplace=True
            )

        if drop_dummy_customer_service_channel:
            data.drop(
                labels='customer_service_channel_Telefone',
                axis=1,
                inplace=True
            )

    if drop_negative_monthly_income:
        data = data.loc[data['monthly_income'] >= 0]

    return data
