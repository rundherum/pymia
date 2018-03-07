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
        d = vars(self)
        for k, v in d.items():
            if isinstance(v, Dictable):
                d[k] = v.to_dict()
        return d

    def from_dict(self, d: dict, **kwargs):
        version = kwargs.get('version', self.version())
        self.handle_version(version, d, **kwargs)

        dict_to_member(self, d)

    def handle_version(self, version, d: dict, **kwargs):
        """ Version handling. To be overwritten if version has to be verified.

        Args:
            version(int): The version
            d: The configuration dict
            **kwargs: Additional arguments
        """
        pass


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


class JSONConfigurationParser:
    """Represents a JSON configuration parser."""

    @staticmethod
    def load(file_path: str, config: ConfigurationBase):
        """Load the configuration file content.

        Args:
            file_path(str): The configuration file path
            config(ConfigurationBase): The configuration instance to be updated
        """
        if not os.path.isfile(file_path):
            raise ValueError('File {} does not exist'.format(file_path))

        d = json.load(file_path)

        meta_dict = d['meta']
        config_dict = d['config']

        meta = MetaData()
        meta.from_dict(meta_dict)

        if meta.type != config.type():
            raise ValueError('configuration type "{}" does not fit the instance type "{}"'.format(
                meta.type, config.__class__.__name__))

        config.from_dict(config_dict, version=meta.version)

    @staticmethod
    def save(file_path: str, config: ConfigurationBase):
        """Save the configuration to file.

        Args:
            file_path: The configuration file path
            config: The configuration
        """
        if not os.path.splitext(file_path)[1] == '.json':
            file_path += '.json'

        meta = MetaData(config.version(), config.type())
        d = {'meta': meta.to_dict(), 'config': config.to_dict()}
        json.dump(d, file_path, sort_keys=True, indent=4)
