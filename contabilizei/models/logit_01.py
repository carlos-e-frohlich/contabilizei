from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from read import import_data, add_regressors

# 1. Import data and add regressors

data = import_data()
data = add_regressors(data)

# 2. Define regressors and regressand

X = data[
    [
        'const',
        # 'age',
        # 'age_squared',
        # 'age_log1p',
        # 'csat',
        # 'csat_squared',
        # 'csat_log1p',
        # 'customer_service_channel_Chat',
        # 'customer_service_channel_Email',
        # 'gender_Feminino',
        'monthly_income',
        'monthly_income_squared',
        # 'monthly_income_log1p',
        # 'n_access_simulator',
        'n_access_simulator_squared',
        # 'n_access_simulator_log1p',
        'n_partners',
        # 'n_partners_squared',
        # 'n_partners_log1p',
        # 'region_Centro-Oeste',
        # 'region_Nordeste',
        # 'region_Norte',
        # 'region_Sul',
        # 'tenure',
        # 'tenure_squared',
        # 'tenure_log1p',
        # 'tickets_opened',
        'tickets_opened_squared',
        # 'tickets_opened_log1p',
        # 'tickets_opened_per_year',
        # 'tickets_opened_per_year_squared',
        # 'tickets_opened_per_year_log1p'
    ]
].astype(float)

y = data[['propension']].astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=.25,
    random_state=966588769 # Randomly selected through random.org.
)

model = sm.Logit(
    endog=y_train,
    exog=X_train
)

results = model.fit()

print(results.summary())
