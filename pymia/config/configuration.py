import abc
import json
import os

import yaml


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
        return member_to_dict(self)

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


def member_to_dict(target_obj):
    d = {}
    for k, v in vars(target_obj).items():
        if isinstance(v, Dictable):
            d[k] = v.to_dict()
        else:
            d[k] = v
    return d


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

    @classmethod
    @abc.abstractmethod
    def load(cls, file_path: str, config_cls):
        """Load the configuration file.

        Args:
            file_path(str): The configuration file path
            config_cls: The type of the configuration

        Returns:
            ConfigurationBase: The configuration
        """
        pass

    @classmethod
    @abc.abstractmethod
    def save(cls, file_path: str, config: ConfigurationBase):
        """Save the configuration to file.

        Args:
            file_path: The configuration file path
            config: The configuration
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def read(file_path):
        """ Read the file content

        Args:
            file_path: The configuration file path

        Returns:
            The file content in the format of the underlying reader

        """

    @staticmethod
    @abc.abstractmethod
    def write(file_path, content):
        """Write the content file content

        Args:
            file_path: The configuration file path
            content: The content to write
        """


class JSONConfigurationParser(ConfigurationParser):
    """Represents a JSON configuration parser."""

    @classmethod
    def load(cls, file_path: str, config_cls):
        if not os.path.isfile(file_path):
            raise ValueError('File {} does not exist'.format(file_path))

        d = cls.read(file_path)

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

    @classmethod
    def save(cls, file_path: str, config: ConfigurationBase):
        meta = MetaData(config.version(), config.type())
        d = {'meta': meta.to_dict(), 'config': config.to_dict()}

        cls.write(file_path, d)

    @staticmethod
    def read(file_path):
        with open(file_path, 'r') as f:
            d = json.load(f)
        return d

    @staticmethod
    def write(file_path, content):
        with open(file_path, 'w') as f:
            json.dump(content, f, sort_keys=True, indent=4)


class YamlConfigurationParser(JSONConfigurationParser):
    """Represents a YAML configuration parser."""

    @staticmethod
    def read(file_path):
        with open(file_path, 'r') as f:
            d = yaml.load(f)
        return d

    @staticmethod
    def write(file_path, content):
        with open(file_path, 'w') as f:
            yaml.dump(content, f, default_flow_style=False)


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


def load_meta(file_path: str) -> MetaData:
    """ Load the meta data of a configuration

    Args:
        file_path: The configuration file path

    Returns:
        MetaData: The meta data of the configuration
    """
    extension = os.path.splitext(file_path)[1]
    if extension not in parser_registry:
        raise ValueError('unknown configuration file extension "{}"'.format(extension))

    d = parser_registry[extension].read(file_path)
    meta_data = MetaData()
    meta_data.from_dict(d['meta'])
    return meta_data


parser_registry = {'.json': JSONConfigurationParser, '.yaml': YamlConfigurationParser, '.yml': YamlConfigurationParser}
