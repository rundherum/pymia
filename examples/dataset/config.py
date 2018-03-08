import miapy.config.configuration as miapy_cfg


class Configuration(miapy_cfg.ConfigurationBase):
    VERSION = 1
    TYPE = ''

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def type(cls) -> str:
        return cls.TYPE

    def __init__(self):
        self.database_file = '/home/fbalsiger/Documents/config.json'


def load(path: str, config_cls):
    """Loads a configuration file.

    Args:
        path (str): The path to the configuration file.
        config_cls (class): The configuration class (not an instance).

    Returns:
        (config_cls): The configuration.
    """

    return miapy_cfg.JSONConfigurationParser.load(path, config_cls)
