import argparse
from dataclasses import dataclass
from typing import List, Literal
import numpy as np
import numpy.random as npr
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
import algos.ppo.core as core
import algos.ppo.ppo as ppo
from algos.ppo.epoch_logger import setup_logger_kwargs, EpochLogger
from gym_rad_search.envs import RadSearch  # type: ignore

@dataclass
class CliArgs:
    selection: str


@dataclass
class TuneArgs:
    hid_gru: int
    hid_pol: int
    hid_val: int
    hid_rec: int
    l_pol: int
    l_val: int
    gamma: float
    seed: int
    steps_per_epoch: int
    epochs: int
    exp_name: str
    dims: tuple[int, int]
    area_obs: tuple[int, int]
    obstruct: Literal[-1, 0, 1]
    net_type: str
    alpha: float
    tuning: bool


def parse_args(parser: argparse.ArgumentParser, rawArgs: list = None) -> CliArgs:
    if rawArgs:
        args, functionArgs = parser.parse_known_args(rawArgs)
        if functionArgs:
            print(f"Warning - Tuning from baselines. Arguments will not be used: {functionArgs}")
    else:
        args, unknown_args = parser.parse_known_args()
        if unknown_args:
            print(f"Warning - Unkown Arguments: {unknown_args}")

    return CliArgs(
        selection=args.selection,
    )


def baseline_args() -> TuneArgs:
    return TuneArgs(
        hid_gru=24,
        hid_pol=32,
        hid_val=32,
        hid_rec=24,
        l_pol=1,
        l_val=1,
        gamma=0.99,
        seed=2,
        steps_per_epoch=480,
        epochs=3000,
        exp_name='default',
        dims=[2700.0, 2700.0],
        area_obs=[200.0, 500.0],
        obstruct=-1,
        net_type="rnn",
        alpha=0.1,
        tuning=True
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=str,
        default="all",
        help="Select hyperparameter to tune (default: all), option: hid_gru, hid_pol,\
            hid_val, hid_rec, l_pol, l_val, gamma, steps_per_epoch, epochs, net_type, alpha"
    )
    return parser


def main(args: list = None):
    if args:
        args = parse_args(create_parser(), args)
    else:
        args = parse_args(create_parser())

    baseline = baseline_args()

    # Change mini-batch size, only been tested with size of 1
    batch_s: int = 1

    # Save directory and experiment name
    env_name: str = "tuning"
    exp_name: str = (
        "loc"
        + str(baseline.hid_rec)
        + "_hid"
        + str(baseline.hid_gru)
        + "_pol"
        + str(baseline.hid_pol)
        + "_val"
        + str(baseline.hid_val)
        + "_"
        + baseline.exp_name
        + f"_ep{baseline.epochs}"
        + f"_steps{baseline.steps_per_epoch}"
    )

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed(baseline.seed))[0]
    rng = npr.default_rng(robust_seed)

    dim_length, dim_height = baseline.dims
    logger_kwargs = setup_logger_kwargs(
        exp_name, baseline.seed, data_dir="../../models/train", env_name=env_name
    )
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    env: RadSearch = RadSearch(
        bbox=np.array(  # type: ignore
            [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
        ),
        area_obs=np.array(baseline.area_obs),  # type: ignore
        obstruct=baseline.obstruct,
        np_random=rng,
    )

    # Run ppo training function
    return ppo.PPO(
        env=env,
        actor_critic=core.RNNModelActorCritic,
        logger=logger,
        ac_kwargs=dict(
            hidden_sizes_pol=[[baseline.hid_pol]] * baseline.l_pol,
            hidden_sizes_val=[[baseline.hid_val]] * baseline.l_val,
            hidden_sizes_rec=[baseline.hid_rec],
            hidden=[[baseline.hid_gru]],
            net_type=baseline.net_type,
            batch_s=batch_s,
        ),
        gamma=baseline.gamma,
        alpha=baseline.alpha,
        seed=robust_seed,
        steps_per_epoch=baseline.steps_per_epoch,
        epochs=baseline.epochs,
        render=False,
        save_gif=False,
    )


if __name__ == "__main__":
    main()  