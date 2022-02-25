

def convert_to_string(data):
    """Converts extracted string data from bytes to string, as strings are handled as bytes since h5py >= 3.0.

    The function has been introduced as part of an `issue <https://github.com/rundherum/pymia/issues/40>`_.

    Args:
        data: The data to be converted; either :obj:`bytes` or list of :obj:`bytes`.

    Returns:
        The converted data as :obj:`str` or list of :obj:`str`.
    """
    if isinstance(data, bytes):
        return data.decode('utf-8')
    elif isinstance(data, list):
        return [convert_to_string(d) for d in data]
    else:
        return data
