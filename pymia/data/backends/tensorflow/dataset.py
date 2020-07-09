import pymia.data.extraction as extr


def get_tf_generator(data_source: extr.PymiaDatasource):
    """Returns a generator that wraps :class:`.PymiaDatasource` for the tensorflow data handling.


    The returned generator can be used with `tf.data.Dataset.from_generator
    <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator>`_ in order to build a tensorflow dataset`_.

    Args:
        data_source (.PymiaDatasource): the datasource to be wrapped.

    Returns:
        generator: Function that loops over the entire datasource and yields all entries.

    """
    def generator():
        for i in range(len(data_source)):
            yield data_source[i]
    return generator
