from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier

from run_pipeline import VTACPipe

vtac = VTACPipe(config_path="../config/config.yaml")

knn = KNeighborsClassifier()
rfc = RandomForestClassifier()
ada = AdaBoostClassifier()

models = {
    # "K-Nearest Neighbors": knn,
    "rfc": rfc,
    # "ada": ada

}
best_model = vtac.train(models)
# print(best_model)
print(vtac)