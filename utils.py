from argparse import Namespace, ArgumentParser


def argparser() -> ArgumentParser:
    parser = ArgumentParser()
    
    parser.add_argument('--version', required=True, type=str)
    
    parser.add_argument('--data_dir', required=True)
    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=4)
    
    parser.add_argument('--lr_plateau', action='store_true')
    
    return parser