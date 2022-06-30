import argparse
import time
from dataclasses import dataclass
from typing import TypedDict, Literal
import numpy as np
import numpy.random as npr
from gym.utils.seeding import _int_list_from_bigint, hash_seed  # type: ignore
import algos.ppo.core as core
import algos.ppo.ppo as ppo
from algos.ppo.epoch_logger import setup_logger_kwargs, EpochLogger
from gym_rad_search.envs import RadSearch  # type: ignore
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.utils import validate_save_restore
import os
import sys

@dataclass
class CliArgs:
    selection: str


#@dataclass
class TuneArgs(TypedDict):
    hid_gru: int
    hid_pol: int
    hid_val: int
    hid_rec: int
    l_pol: int
    l_val: int
    gamma: float
    steps_per_epoch: int
    epochs: int
    exp_name: str
    dims: tuple[int, int]
    area_obs: tuple[int, int]
    obstruct: Literal[-1, 0, 1]
    net_type: str
    alpha: float
    batch_s: int

#@dataclass
class EnvArgs(TypedDict):
    #env: RadSearch # TODO Visilibity lib cannot be pickled/serialized, must generate env on thread of execution
    #logger: EpochLogger # TODO _io.TextIOWrapper' object cannot be serialized, must generate env on thread of execution
    model_dir: str
    model_title: str
    exp_name: str
    seed: int
    render: bool
    save_gif: bool
    tuning: bool
    hyperparameters: TuneArgs


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


def test_cluster():
    def run_me(config):
        for iter in range(10):
            time.sleep(1)
            tune.report(hello="world", ray="tune")

    start = time.time()
    analysis = tune.run(run_me)
    taken = time.time() - start

    print(f"Time: {taken:.2f} seconds.")
    print("Best hyperparameters found were: ", analysis.get_best_config(metric="mean_loss", mode="min"))
    print("Best result: ", analysis.get_best_trial(metric="mean_loss", mode="min"))


def generate_baseline_args() -> TuneArgs:
    return TuneArgs(
        hid_gru=24,
        hid_pol=32,
        hid_val=32,
        hid_rec=24,
        l_pol=1,
        l_val=1,
        gamma=0.99,
        steps_per_epoch=480,
        epochs=3000,
        exp_name='default',
        dims=[2700.0, 2700.0],
        area_obs=[200.0, 500.0],
        obstruct=-1,
        net_type="rnn",
        alpha=0.1,
        batch_s=1
    )


def generate_env_args() -> EnvArgs:
    baseline = generate_baseline_args()

    # Save directory and experiment name
    model_dir: str = "models/train"
    model_title: str = "tuning"
    id = os.getpid()
    exp_name: str = (
        "loc"
        + str(baseline['hid_rec'])
        + "_hid"
        + str(baseline['hid_gru'])
        + "_pol"
        + str(baseline['hid_pol'])
        + "_val"
        + str(baseline['hid_val'])
        + "_"
        + baseline['exp_name']
        + f"_ep{baseline['epochs']}"
        + f"_steps{baseline['steps_per_epoch']}"
        + f"_id{id}"
    )

    return EnvArgs(
        #env=env, # TODO Visilibity lib cannot be pickled/serialized, must generate env on thread of execution
        #logger=logger, # TODO Visilibity lib cannot be pickled/serialized, must generate env on thread of execution
        seed=0,
        model_title=model_title,
        exp_name=exp_name,
        model_dir=model_dir,
        render=False,
        save_gif=False,
        tuning=True,
        hyperparameters=baseline,
    )


def create_search_space(args: EnvArgs) -> EnvArgs:
    args['hyperparameters']['epochs'] = tune.qrandint(lower=1, upper=10, q=1).sample()
    return args


def objective(config, checkpoint_dir=None):
    # Mute stdout for thread
    stdout_save = sys.stdout
    sys.stdout = open('/dev/null', 'w')

    # Generate a large random seed and random generator object for reproducibility
    robust_seed = _int_list_from_bigint(hash_seed(config['seed']))[0]
    rng = npr.default_rng(robust_seed)

    # Set up logger and save configuration
    logger_kwargs = setup_logger_kwargs( 
        exp_name=config['exp_name'], seed=config['seed'], data_dir=config['model_dir'], env_name=config['model_title']
    )
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Set up env # TODO Visilibity lib cannot be pickled/serialized, must generate env on thread of execution
    dim_length, dim_height = config['hyperparameters']['dims']
    env: RadSearch = RadSearch(
        bbox=np.array(  # type: ignore
            [[0.0, 0.0], [dim_length, 0.0], [dim_length, dim_height], [0.0, dim_height]]
        ),
        area_obs=np.array(config['hyperparameters']['area_obs']),  # type: ignore
        obstruct=config['hyperparameters']['obstruct'],
        np_random=rng,
    )

    model = ppo.PPO(
        env=env,
        actor_critic=core.RNNModelActorCritic,
        logger=logger,
        ac_kwargs=dict(
            hidden_sizes_pol=[[config['hyperparameters']['hid_pol']]] * config['hyperparameters']['l_pol'],
            hidden_sizes_val=[[config['hyperparameters']['hid_val']]] * config['hyperparameters']['l_val'],
            hidden_sizes_rec=[config['hyperparameters']['hid_rec']],
            hidden=[[config['hyperparameters']['hid_gru']]],
            net_type=config['hyperparameters']['net_type'],
            batch_s=config['hyperparameters']['batch_s'],
        ),
        gamma=config['hyperparameters']['gamma'],
        alpha=config['hyperparameters']['alpha'],
        seed=config['seed'],
        steps_per_epoch=config['hyperparameters']['steps_per_epoch'],
        epochs=config['hyperparameters']['epochs'],
        render=config['render'],
        save_gif=config['save_gif'],
        tuning=config['tuning']
    )
    result = model.train()

    # Reset stdout
    sys.stdout = stdout_save

    tune.report(fitness=result)


def main(args: list = None):
    # Connect to driver node
    ray.init(address='auto')

    args = parse_args(create_parser(), args) if args else parse_args(create_parser())
    search_space = create_search_space(generate_env_args())

    # Run ppo training function
    print('\n~~~Beginning training...~~~')
    start = time.time()
    # DEBUG
    objective(config=search_space)

    # AsyncHyperBand enables aggressive early stopping of bad trials.
    scheduler = AsyncHyperBandScheduler(
        grace_period=5,
        metric='fitness',
        mode='max',
        reduction_factor=3,
        brackets=1,
        stop_last_trials=False,
        )

    # Begin tuning
    analysis = tune.run(
        objective, 
        config=search_space, 
        scheduler=scheduler, 
        #checkpoint_freq=1, 
        num_samples=10,
        )

    taken = time.time() - start
    print(f"Time: {taken:.2f} seconds.")
    print("Best hyperparameters found were: ", analysis.get_best_config(metric="fitness", mode="max"))
    print("Best result: ", analysis.get_best_trial(metric="fitness", mode="max"))

    return


if __name__ == "__main__":
    main()  