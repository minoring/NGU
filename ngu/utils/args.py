import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Never Give Up')
    parser.add_argument('--env',
                        help='Environment ID to create (default: MontezumaRevengeNoFrameskip-v4)',
                        default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--num-actors',
                        type=int,
                        default=32,
                        help='The number of parallel actor to collect experience (default: 32)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--no-gpu', action='store_true', help='Whether not to use gpu')
    parser.add_argument('--log-interval',
                        type=int,
                        default=200,
                        help='Logging interval in terms of the number of steps (default: 100)')
    args = parser.parse_args()

    return args
