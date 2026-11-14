"""Build the three feature indexes used by SoftVcStylizer inference."""
import argparse
from pathlib import Path
import pickle

from frontend import FEATURE_STREAMS, options
from logger import utils


def parse_args(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--output_dir", "--index_dir", dest="output_dir", default=None)
    parser.add_argument("--speaker_id", type=int, default=1)
    parser.add_argument("--speaker_subdir", default="", help="Optional speaker directory below each feature stream")
    return parser.parse_args(args)


def main():
    command = parse_args()
    args = utils.load_config(command.config) if command.config else utils.DotDict({})
    destination = command.output_dir or options(args).get("index_dir")
    if not destination:
        raise ValueError("Set --output_dir or frontend.index_dir in --config")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = command.root_dir or "data"
    paths = {}
    if command.config and command.root_dir is None:
        paths = dict(train_path=args.data.train_path, valid_path=args.data.valid_path)
    from feature_retrival.utils import train_index

    for stream, filename in FEATURE_STREAMS.items():
        index = train_index(command.speaker_subdir, root, feat=stream, **paths)
        with (destination / filename).open("wb") as file:
            pickle.dump({command.speaker_id: index}, file)
        print(f"Saved {stream}: {destination / filename}")


if __name__ == "__main__":
    main()
