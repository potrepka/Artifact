from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import common.data as data
import os
from tqdm import tqdm
import zipfile

def main(params):
    with zipfile.ZipFile(data.file) as zf:
        for member in tqdm(zf.infolist(), desc='Extracting COCO'):
            try:
                zf.extract(member, os.path.dirname(data.file))
            except zipfile.error as e:
                pass

    parent = os.path.dirname(data.directory)
    old_name = os.path.splitext(os.path.basename(data.url))[0]
    old_directory = os.path.join(parent, old_name)
    os.rename(old_directory, data.directory)

    os.remove(data.file)

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    params = parser.parse_args()
    main(params)
