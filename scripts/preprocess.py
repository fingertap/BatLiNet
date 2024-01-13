import os
import argparse
import importlib

from tqdm import tqdm
from pathlib import Path


def import_function(path: Path or str) -> callable:
    # convert the path to a string if it is a pathlib.Path object
    if isinstance(path, Path):
        path = str(path)
    # get the file name without the extension
    filename = Path(path).stem
    # create a module specification from the file path
    spec = importlib.util.spec_from_file_location(filename, path)
    # create a module object from the specification
    module = importlib.util.module_from_spec(spec)
    # execute the module code
    spec.loader.exec_module(module)
    # get the function object from the module
    function = module.__dict__.get('preprocess')
    # return the function object
    return function


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BatLiNet data preprocessing CLI tool.')
    parser.add_argument('--input-path', dest='input_path', required=True)
    parser.add_argument('--output-path', dest='output_path', required=True)

    args = parser.parse_args()

    script_path = Path(__file__).parent / 'preprocess_scripts'
    raw_data_path = Path(args.input_path)
    processed_data_path = Path(args.output_path)

    pbar = tqdm([
        raw_data_path / 'SNL',
        raw_data_path / 'UL_PUR',
        raw_data_path / 'HNEI',
        raw_data_path / 'MATR',
        raw_data_path / 'HUST',
        raw_data_path / 'RWTH',
        raw_data_path / 'CALCE',
    ])

    for path in pbar:
        if not path.exists():
            continue

        dataset = path.stem
        store_dir = processed_data_path / dataset

        if not store_dir.exists():
            os.makedirs(store_dir)

        # Already processed
        if len(list(store_dir.glob('*.pkl'))) > 0:
            continue

        pbar.set_description(f'Processing dataset {dataset}')

        preprocess = import_function(script_path / f'preprocess_{dataset}.py')
        
        for cell in preprocess(path):
            store_path = store_dir / f'{cell.cell_id}.pkl'
            cell.dump(store_path)
