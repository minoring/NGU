from ngu.utils.args import get_args
from ngu.utils.random_util import set_global_seed
from ngu.utils.pytorch_util import init_device


def main():
    args = get_args()

    set_global_seed(args.seed)
    init_device()


if __name__ == '__main__':
    main()
