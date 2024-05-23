import yaml

config_path = "../config/config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_dict = []
for model in config['models']:
    print(model)
    model_name = model
    model_class = model['class']
    print(model_class)
    # model_params = model['param_grid']
    # model_dict.append((model_name, model_class, model_params))