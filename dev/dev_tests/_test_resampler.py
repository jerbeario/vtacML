import pandas as pd
from imblearn.over_sampling import SMOTE

data = pd.read_parquet('/vtacML/data/combined_qpo_vt_with_GRB.parquet')
print(data['IS_GRB'].value_counts())
columns = ["MAGCAL_R0",
           "MAGCAL_B0",
           "MAGERR_R0",
           "MAGERR_B0",
           "MAGCAL_R1",
           "MAGCAL_B1",
           "MAGERR_R1",
           "MAGERR_B1",
           "MAGVAR_R1",
           "MAGVAR_B1",
           #    "CASE",
           #    "EFLAG_B1"
           "IS_GRB"
           ]

data = data[columns].dropna()


def resample(X):
    sm = SMOTE(sampling_strategy='minority', random_state=42)
    X_, y_ = sm.fit_resample(X.drop('IS_GRB', axis=1), X['IS_GRB'])
    X_resampled = pd.concat([pd.DataFrame(y_), pd.DataFrame(X_)], axis=1)
    return X_resampled


resampled_data = resample(data)
print(resampled_data['IS_GRB'].value_counts())
