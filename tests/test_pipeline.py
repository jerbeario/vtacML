from sklearn.ensemble import RandomForestClassifier
import pytest
from vtacML.pipeline import VTACMLPipe
from vtacML import get_path
import pandas as pd

data_path = get_path('/data/combined_qpo_vt_all_cases_with_GRB_with_flags.parquet')
test_df = pd.read_parquet(data_path)
def test_vtacMLPipe_init():
    vtacML_pipe = VTACMLPipe(config_file='config/test_config.yaml')
    assert vtacML_pipe.config is not None
    assert vtacML_pipe.preprocessing is not None

def test_vtacMLPipe_train():
    vtacML_pipe = VTACMLPipe(config_file='config/test_config.yaml')
    assert vtacML_pipe.best_model is None
    vtacML_pipe.train(cv=2)
    assert vtacML_pipe.best_model is not None


def test_vtacMLPipe_save_model():
    vtacML_pipe = VTACMLPipe(config_file='config/test_config.yaml')
    vtacML_pipe.train(cv=2)
    vtacML_pipe.save_best_model(model_name='test_model.pkl')
    assert vtacML_pipe.best_model.steps[-1][0] == 'rfc'

def test_vtacMLPipe_load_model():
    vtacML_pipe = VTACMLPipe(config_file='config/test_config.yaml')
    vtacML_pipe.load_best_model(model_name='test_model.pkl')
def test_vtacMLPipe_predict():
    vtacML_pipe = VTACMLPipe(config_file='config/test_config.yaml')
    vtacML_pipe.load_best_model(model_name='test_model.pkl')
    test_X = test_df.drop('IS_GRB', axis=1)
    test_y = test_df['IS_GRB']
    pred_y = vtacML_pipe.predict(test_X, prob=False)
    assert len(pred_y) == len(test_y)

