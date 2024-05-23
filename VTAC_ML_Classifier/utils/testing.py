from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import logging
import pandas as pd
from run_pipeline import VTACMLPipe

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
#

config_path = "../config/config.yaml"
faint_case_path = "/Users/jeremypalmerio/Repos/VTAC_ML/data/combined_qpo_vt_faint_case_with_GRB_with_flags.parquet"
# testing default vtac_ml
vtac_ml_default = VTACMLPipe(config_path=config_path)

print(vtac_ml_default.preprocessing)
knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()
models = {
    # "K-Nearest Neighbors": knn,
    "rfc": rfc,
    # "ada": ada

}
# vtac_ml_default.train(models=models, resample=False)
# vtac_ml_default.save_best_model()
vtac_ml_default.load_best_model()
# vtac_ml_default.evaluate()
vtac_ml_default.evaluate()






# vtac_ml_default.load_best_model()
# vtac_ml_default.predict()
# vtac_ml_default.load_data(data, columns, target)
# vtac_ml_default.train(models)




# vtac_ml_pipe = VTACMLPipe(config_path="../config/config.yaml")
#
# knn = KNeighborsClassifier()
# rfc = RandomForestClassifier()
# ada = AdaBoostClassifier()
#
# model_path = '../output/best_model'
#
# models = {
#     # "K-Nearest Neighbors": knn,
#     "rfc": rfc,
#     "ada": ada
#
# }
# vtac_ml_pipe.load_best_model(model_path=model_path)


# print(vtac_ml_pipe.X_train, vtac_ml_pipe.y_train)
# best_model = vtac_ml_pipe.train(models, model_path=model_path)
# vtac_ml_pipe.evaluate()
# # print(best_model)
# print(vtac_ml_pipe)

