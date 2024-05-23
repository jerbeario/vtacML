from run_pipeline import VTACMLPipe

config_path = "../config/config.yaml"
vtac_ml = VTACMLPipe(config_path=config_path)

vtac_ml.load_best_model()
vtac_ml.evaluate(title='rfc_ada_knn_FullGridSearch')