import abc
import json
import os


class Dictable(metaclass=abc.ABCMeta):
    """Interface allowing dict serialization of an object."""

    @abc.abstractmethod
    def to_dict(self, **kwargs):
        """
        Transform instance to a dict

        Args:
            **kwargs: Additional arguments

        Returns:
            dict: Instance variables in dict
        """
        pass

    @abc.abstractmethod
    def from_dict(self, d: dict, **kwargs):
        """
        Updated instance by a dict

        Args:
            d(dict): Dict containing variables to be updated
            **kwargs: Additional arguments
        """
        pass


class ConfigurationBase(Dictable, metaclass=abc.ABCMeta):
    """Base class for Configurations."""
    @classmethod
    @abc.abstractmethod
    def version(cls) -> int:
        """Configuration version property"""
        pass

    @classmethod
    @abc.abstractmethod
    def type(cls) -> str:
        """Configuration type property"""
        pass

    def to_dict(self, **kwargs):
        d = {}
        for k, v in vars(self).items():
            if isinstance(v, Dictable):
                d[k] = v.to_dict()
            else:
                d[k] = v
        return d

    def from_dict(self, d: dict, **kwargs):
        version = kwargs.get('version', self.version())
        self.handle_version(version, d)

        dict_to_member(self, d)

    def handle_version(self, version, d: dict):
        """ Version handling. To be overwritten if version has to be verified.

        Args:
            version(int): The version
            d: The configuration dict
        """
        pass

    def __str__(self):
        """Gets a printable string representation.

        Returns:
            str: String representation.
        """
        dict_as_string = str(self.to_dict())
        dict_as_string = dict_as_string.replace('{', '').replace('}', '').replace(',', '\n')
        return '{}:\n' \
               ' {}\n' \
            .format(type(self).__name__, dict_as_string)


def combine_dicts(target, source):
    for key, value in source.items():
        target[key] = value


def dict_to_member(target_obj, d: dict):
    members = vars(target_obj)  # vars has direct reference to writable members
    for key in members:
        if key in d:
            if isinstance(members[key], dict):
                combine_dicts(members[key], d[key])
            elif isinstance(members[key], Dictable):
                members[key].from_dict(d[key])
            else:
                members[key] = d[key]


class MetaData(Dictable):
    """Represents the meta-information in a configuration file."""
    def __init__(self, version: int=-1, type_: str='') -> None:
        self.version = version
        self.type = type_

    def to_dict(self, **kwargs):
        return vars(self)

    def from_dict(self, d: dict, **kwargs):
        self.version = d.get('version')
        self.type = d.get('type')


class ConfigurationParser(metaclass=abc.ABCMeta):
    """Represents the configuration parser interface."""

    @staticmethod
    @abc.abstractmethod
    def load(file_path: str, config_cls):
        """Load the configuration file.

        Args:
            file_path(str): The configuration file path
            config_cls: The type of the configuration

        Returns:
            ConfigurationBase: The configuration
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def save(file_path: str, config: ConfigurationBase):
        """Save the configuration to file.

        Args:
            file_path: The configuration file path
            config: The configuration
        """
        pass


class JSONConfigurationParser(ConfigurationParser):
    """Represents a JSON configuration parser."""

    @staticmethod
    def load(file_path: str, config_cls):
        if not os.path.isfile(file_path):
            raise ValueError('File {} does not exist'.format(file_path))

        with open(file_path, 'r') as f:
            d = json.load(f)

        meta_dict = d['meta']
        config_dict = d['config']

        meta = MetaData()
        meta.from_dict(meta_dict)

        config = config_cls()
        if meta.type != config.type():
            raise ValueError('configuration type "{}" does not fit the configuration instance "{}"'.format(
                meta.type, config.__class__.__name__))

        config.from_dict(config_dict, version=meta.version)
        return config

    @staticmethod
    def save(file_path: str, config: ConfigurationBase):
        meta = MetaData(config.version(), config.type())
        d = {'meta': meta.to_dict(), 'config': config.to_dict()}

        with open(file_path, 'w') as f:
            json.dump(d, f, sort_keys=True, indent=4)


def load(file_path: str, config_cls) -> ConfigurationBase:
    """Load the configuration file.

    Args:
        file_path(str): The configuration file path
        config_cls: The configuration class type

    Returns:
        ConfigurationBase: The configuration
    """
    extension = os.path.splitext(file_path)[1]
    if extension not in parser_registry:
        raise ValueError('unknown configuration file extension "{}"'.format(extension))

    return parser_registry[extension].load(file_path, config_cls)


def save(file_path: str, config: ConfigurationBase) -> None:
    """Save the configuration to file.

    Args:
        file_path: The configuration file path
        config: The configuration
    """

    extension = os.path.splitext(file_path)[1]
    if extension not in parser_registry:
        raise ValueError('unknown configuration file extension "{}"'.format(extension))

    return parser_registry[extension].save(file_path, config)


parser_registry = {'.json': JSONConfigurationParser}



