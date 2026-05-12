import importlib
from easydict import EasyDict


class ParamManager:
    
    def __init__(self, args):
        
        self.args = EasyDict(dict(vars(args)))   
        
def add_config_param(old_args, config_file_name = None):
    
    if config_file_name is None:
        config_file_name = old_args.config_file_name
        
    if config_file_name.endswith('.py'):
        module_name = '.' + config_file_name[:-3]
    else:
        module_name = '.' + config_file_name

    config = importlib.import_module(module_name, 'configs')

    config_param = config.Param
    method_args = config_param(old_args)

    cli_override_keys = set(old_args.get('_cli_overrides', []))
    new_args = dict(old_args)

    for key, value in method_args.hyper_param.items():
        if key not in cli_override_keys:
            new_args[key] = value

    new_args = EasyDict(new_args)
    
    return new_args
