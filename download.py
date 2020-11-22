from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import common.data as data
import os
import requests
from tqdm import tqdm

def main(params):
    request = requests.get(data.url, stream=True)
    os.makedirs(os.path.dirname(data.file), exist_ok=True)
    with open(data.file, 'wb') as f:
        length = int(request.headers.get('content-length'))
        chunk_size = 512 * 1024
        t = tqdm(total=length, unit='B', unit_scale=True, desc='Downloading COCO')
        for chunk in request.iter_content(chunk_size=chunk_size):
            if chunk:
                t.update(chunk_size)
                f.write(chunk)
                f.flush()

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    params = parser.parse_args()
    main(params)
