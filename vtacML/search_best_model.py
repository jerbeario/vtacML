from .pipeline import VTACMLPipe

if __name__ == "__main__":
    vtac_ml = VTACMLPipe()

    vtac_ml.train()
    vtac_ml.save_best_model()