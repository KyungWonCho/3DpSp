from argparse import ArgumentParser


class TestOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.parser.add_argument('--exp_dir', type=str, required=True, help='Path to experiment output directory')
        self.parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to model checkpoint')
        self.parser.add_argument('--data_path', type=str, required=True, help='Path to data directory')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
