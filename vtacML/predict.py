""" Script to predict trained vtacML models """

import pandas as pd
from .pipeline import VTACMLPipe

if __name__ == "__main__":
    vtac_ml = VTACMLPipe()
    test_data = pd.read_parquet(
        "vtacML/data/combined_qpo_vt_all_cases_with_GRB_with_flags.parquet"
    ).head(1)

    vtac_ml.load_best_model("0.893_rfc_best_model.pkl")
    predictions = vtac_ml.predict(test_data)
    print(predictions)
    print(sum(predictions))
    print(len(predictions))
