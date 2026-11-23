import tempfile
import unittest
from pathlib import Path

from pymia.evaluation.evaluator import Result
from pymia.evaluation.writer import CSVStatisticsWriter, CSVWriter


class TestCSVWriters(unittest.TestCase):

    def test_add_csv_extension_to_filename(self):
        results = [Result('sample', 'label', 'metric', 1.0)]
        for writer in (CSVWriter, CSVStatisticsWriter):
            for extension in ('', '.CSV'):
                with self.subTest(writer=writer.__name__, extension=extension):
                    with tempfile.TemporaryDirectory() as directory:
                        path = Path(directory) / f'results{extension}'
                        writer(str(path)).write(results)
                        expected = Path(directory) / f'results{extension or ".csv"}'
                        self.assertTrue(expected.is_file())

    def test_existing_directory_preserves_output_location(self):
        results = [Result('sample', 'label', 'metric', 1.0)]
        for writer in (CSVWriter, CSVStatisticsWriter):
            with self.subTest(writer=writer.__name__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'results'
                    path.mkdir()
                    sibling = path.with_suffix('.csv')
                    sibling.write_text('existing output', encoding='utf-8')

                    writer(str(path)).write(results)

                    self.assertTrue((path / '.csv').is_file())
                    self.assertEqual(sibling.read_text(encoding='utf-8'), 'existing output')
