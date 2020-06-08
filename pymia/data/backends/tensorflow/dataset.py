
import pymia.data.extraction as extr


# todo(fabian): does not even require tensorflow, still need package?
def get_tf_generator(data_source: extr.PymiaDatasource):
    def generator():
        for i in range(len(data_source)):
            yield data_source[i]
    return generator
