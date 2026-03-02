import yaml

class AttrDict(dict):
    """允许用点语法访问字典的类（如 config.data.train_path）"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_config(config_path):
    """加载并返回配置字典"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return AttrDict(config)