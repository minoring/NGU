import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Never Give Up!')
    parser.add_argument('--env-id',
                        help='Environment ID to create (default: MontezumaRevengeNoFrameskip-v4)',
                        default='MontezumaRevengeNoFrameskip-v4')
    parser.add_argument('--n-actors',
                        type=int,
                        default=64,
                        help='The number of parallel actor to collect experience (default: 64)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed (default: 1)')
    parser.add_argument('--no-gpu', action='store_true', help='Whether not to use gpu')
    parser.add_argument(
        '--model-save-interval',
        type=int,
        help='How many learning steps before saving the trained model. (default: 1000)',
        default=1000)
    parser.add_argument('--video-save-interval',
                        type=int,
                        help='How many steps to train before save the video.')
    args = parser.parse_args()

    return args
