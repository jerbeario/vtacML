from vtacML.pipeline import predict_from_best_pipeline
import pandas as pd
import os

cdir = os.getcwd()
print(cdir)
config_path = "../../vtacML/config/config.yaml"

data = pd.read_parquet(f'{cdir}/vtacML/data/combined_qpo_vt_all_cases_with_GRB_with_flags.parquet')
print(data.head())
X_GRB = data[data['IS_GRB'] == 1]

y = predict_from_best_pipeline(X=X_GRB)
#
# print(sum(y), len(y))


