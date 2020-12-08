#!/usr/bin/env python3
# this file was copied from: https://github.com/josepablocam/ams/tree/master/experiments and adapted for openml data fetching

from argparse import ArgumentParser
import os

import pmlb
import openml

DATASETS_PMLB = [
    "Hill_Valley_without_noise",
    "Hill_Valley_with_noise",
    "breast-cancer-wisconsin",
    "car-evaluation",
    "glass",
    "ionosphere",
    "spambase",
    "wine-quality-red",
    "wine-quality-white",
]

DATASETS_OPENML = {
    'iris': 61,
    'airlines': 1169,
}

DEFAULT_LOCAL_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data/benchmarks-datasets/")
)
# overwrite this if $DATA environment variable set
if "DATA" in os.environ:
    data_dir = os.environ["DATA"]
    cache_dir = os.path.join(data_dir, "benchmarks-datasets")
    DEFAULT_LOCAL_CACHE_DIR = cache_dir

print(
    "Setting default benchmarks-datasets cache directory to: {}".
        format(DEFAULT_LOCAL_CACHE_DIR)
)

openml.config.cache_directory = DEFAULT_LOCAL_CACHE_DIR


def get_args():
    parser = ArgumentParser(description="Download datasets to local cache")
    parser.add_argument("--output", type=str, help="Path to local cache")
    parser.add_argument("--pmlb", type=bool, help="Download default PMLB datasets")
    parser.add_argument("--openml", type=bool, help="Download default openML datasets")
    parser.add_argument("--target", type=str, help="Target attribute")
    return parser.parse_args()


def get_openml_data(id, arg_target = None):
    data = openml.datasets.get_dataset(id, download_data=False)
    target = data.default_target_attribute
    if not arg_target is None:
        target = arg_target
    if (target == None or ',' in target):  # can be empty or contain multiple comma-separated targets
        raise ImportError('One and only one target attribute must be supplied. Please set target environment variable.')
    else:
        dataset = openml.datasets.get_dataset(id, download_data=True)
        return dataset.get_data(target=target, dataset_format="array") + (dataset.qualities.get('NumberOfClasses', 0), )#(len(dataset.retrieve_class_labels(target_name=dataset.default_target_attribute)), )

def main():
    global args
    args = get_args()
    local_cache_dir = DEFAULT_LOCAL_CACHE_DIR
    if args.output is not None:
        local_cache_dir = args.output
    if not os.path.exists(local_cache_dir):
        print("Creating", local_cache_dir)
        os.makedirs(local_cache_dir, exist_ok=True)

    if args.pmlb:
        for d in DATASETS_PMLB:
            print("Downloading", d)
            pmlb.fetch_data(
                d,
                return_X_y=True,
                local_cache_dir=local_cache_dir,
            )
    if args.openml or True:
        for data, id in DATASETS_OPENML.items():
            print("Downloading", data)
            get_openml_data(id, args.target)


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        import pdb

        pdb.post_mortem()
