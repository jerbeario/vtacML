from sklearn.metrics import f1_score

from run_pipeline import predict_from_best_pipeline
import pandas as pd
config_path = "../config/config.yaml"

data = pd.read_parquet('~/Repos/VTAC_ML/data/combined_qpo_vt_with_GRB.parquet')
data_faint = pd.read_parquet('/Users/jeremypalmerio/Repos/VTAC_ML/data/combined_qpo_vt_faint_case_with_GRB_with_flags.parquet')
X_GRB = data[data['IS_GRB'] == 1]

y = predict_from_best_pipeline(X=X_GRB, config_file=config_path)

print(sum(y), len(y))


