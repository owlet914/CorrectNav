import argparse
import gzip
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json_gz', type=str, required=True, help='Input VLN-CE JSON.GZ file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--split', type=str, required=True, help='Dataset split name')
    parser.add_argument('--n_part', type=int, required=True, help='Number of output parts')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with gzip.open(args.input_json_gz, 'rt') as f:
        dataset = json.load(f)

    episodes = dataset['episodes']
    if args.n_part < 1 or args.n_part > len(episodes):
        raise ValueError('n_part must be between 1 and the episode count')

    chunk_size = len(episodes) // args.n_part
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for part_idx in range(1, args.n_part + 1):
        start_idx = (part_idx - 1) * chunk_size
        end_idx = len(episodes) if part_idx == args.n_part else part_idx * chunk_size
        with gzip.open(output_dir / f"{args.split}_part{part_idx}.json.gz", 'wt') as f:
            json.dump({**dataset, 'episodes': episodes[start_idx:end_idx]}, f)
