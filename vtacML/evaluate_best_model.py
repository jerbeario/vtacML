""" Evaluate best model given a trained model. """

from .pipeline import VTACMLPipe

if __name__ == "__main__":
    vtac_ml = VTACMLPipe()

    vtac_ml.load_best_model(model_name="0.864_rfc_best_model.pkl")
    vtac_ml.evaluate(name="test")
