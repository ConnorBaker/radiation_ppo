from algos.ppo import main as ppo_main
from tuning.hypertune import main as hypertune
import argparse
from dataclasses import dataclass


@dataclass
class CliArgs:
    command: str
    functionArgs: list


def parse_args(parser: argparse.ArgumentParser) -> CliArgs:
    args, functionArgs = parser.parse_known_args()
    return CliArgs(
        command = args.command, 
        functionArgs=functionArgs
        )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--command",
        type=str,
        default="help",
        help="Command to run, options: train, test, tune, help",
    )
    return parser


if __name__ == "__main__":
    args = parse_args(create_parser())
    match args.command:
        case 'train':
            ppo_main.main(args.functionArgs)
        case 'test':
            print('Coming soon...')
        case 'tune':
            hypertune(args.functionArgs)
        case 'help':
            print('Type run.py --help for commands list')
        case unknown_command:
            print (f"Unknown command '{unknown_command}'")

    print('Complete.')