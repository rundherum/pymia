import argparse
import io
import os
import urllib.request as request
import zipfile


def main(url, out_dir):
    # download the data
    print(f'Downloading... ({url})')
    resp = request.urlopen(url)
    zip_ = zipfile.ZipFile(io.BytesIO(resp.read()))

    print(f'Extracting... (to {out_dir})')
    members = zip_.infolist()
    for member in members:
        if member.filename.startswith('pymia-example-data-master/example-data/Subject_'):
            if not os.path.basename(member.filename):
                # is a directory
                continue
            member.filename = member.filename.replace('pymia-example-data-master/example-data/', '')
            zip_.extract(member, out_dir)
            print(f'extract {os.path.join(out_dir, member.filename)}')
    print('Finished')


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """

    parser = argparse.ArgumentParser(description='Pull example data')

    parser.add_argument(
        '--url',
        type=str,
        default='https://github.com/rundherum/pymia-example-data/archive/master.zip',
        help='Path to the example data zip file.'
    )

    parser.add_argument(
        '--out_dir',
        type=str,
        default='.',
        help='Path to extract the data to.'
    )

    args = parser.parse_args()
    main(args.url, args.out_dir)
