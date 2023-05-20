import importlib


def get_setting_func(setting):
    assert len(setting.split(':')) == 2, \
            'Setting should be "script_path:func_name"'
    script_path, func_name = setting.split(':')
    assert script_path.endswith('.py'), \
            'Script should end with ".py"'
    module_name = script_path[:-3].replace('/', '.')
    while module_name.startswith('.'):
        module_name = module_name[1:]
    try:
        load_setting_module = importlib.import_module(module_name)
    except:
        module_name = 'babylm_baseline_train.configs.' + module_name
        load_setting_module = importlib.import_module(module_name)
    setting_func = getattr(load_setting_module, func_name)
    return setting_func
