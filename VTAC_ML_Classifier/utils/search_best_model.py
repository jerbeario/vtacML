from run_pipeline import VTACMLPipe

config_path = "../config/config.yaml"
vtac_ml = VTACMLPipe(config_path=config_path)

vtac_ml.train()
vtac_ml.save_best_model()


