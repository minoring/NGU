import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Never Give Up')
    parser.add_argument('--env',
                        help='Environment ID to create (default: MontezumaRevengeNoFrameskip-v4)',
                        default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument(
        '--num-env',
        type=int,
        default=1,
        help='The number of parallel environment to collect experience (default: 1)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--no-gpu', action='store_true', help='Whether not to use gpu')
    args = parser.parse_args()

    return args
