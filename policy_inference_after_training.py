import numpy as np

import ray
from ray import tune
from ray.tune.analysis.experiment_analysis import ExperimentAnalysis
from ray.tune.stopper import (
    TrialPlateauStopper,
    CombinedStopper,
    MaximumIterationStopper,
)
from ray.tune import CLIReporter
from ray.tune.syncer import SyncConfig
from ray.tune.tune_config import TuneConfig
from ray.air import Checkpoint
from ray.air.config import RunConfig
from ray.train.rl.rl_trainer import RLTrainer
from ray.air.result import Result
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ddppo import DDPPOConfig
from gym_rad_search.envs.rad_search_env import (
    Interval,
    RadSearch,
    RadSearchConfig,
    mk_rectangle,
)
from ray.tune.tuner import Tuner


def train() -> ExperimentAnalysis:
    env_config = RadSearchConfig(
        observation_area=Interval(np.array((200.0, 500.0))),
        bounding_box=mk_rectangle(2700.0, 2700.0),
        obstruction_setting=1,
        seed=0,
    )

    ddppo_config: DDPPOConfig = (
        DDPPOConfig()  # type: ignore
        .framework("torch")
        .rollouts(
            num_rollout_workers=1,
            rollout_fragment_length=128,
            num_envs_per_worker=1,
            horizon=128,
        )
        .resources(num_gpus_per_worker=0)
        .environment(env="RadSearch-v1", env_config=env_config)
    )

    stopper = CombinedStopper(
        TrialPlateauStopper("episode_reward_mean", std=0.001),
        TrialPlateauStopper("episode_len_mean", std=0.01),
        MaximumIterationStopper(750),
    )

    t1 = tune.run(
        # Trainable
        run_or_experiment=RLTrainer(
            algorithm="DDPPO", config=ddppo_config.to_dict()  # type: ignore
        ).as_trainable(),
        # TuneConfig
        metric="episode_reward_mean",
        mode="max",
        num_samples=5,
        # We can specify a local directory to avoid polluting a particular disk.
        # local_dir="~/ramdisk/ray_results",
        # RunConfig
        name="blarg2",
        stop=stopper,
        verbose=0,
        # Hyperparameters
        config={
            "num_envs_per_worker": tune.grid_search(range(1, 9)),
            "rollout_fragment_length": tune.grid_search([2 ** (7+i) for i in range(1, 21)]),
        },
        checkpoint_freq=300,
        checkpoint_at_end=True,
        max_failures=-1,
        # sync_config=tune.SyncConfig(
            # upload_dir="gs://ray-blarg-test-bucket",
            # Custom sync command for s3-like endpoints
            # syncer="aws s3 sync {source} {target} --endpoint-url https://b10fa25202b183e3807763a0b0320d47.r2.cloudflarestorage.com",  
        # ),
        resume="AUTO",
    )

    print(f"Best trial: {t1.best_trial}")
    print(f"Best config: {t1.best_config}")
    print(f"Best checkpint: {t1.best_checkpoint}")

    return t1


if __name__ == "__main__":
    ray.init(address="auto")
    tune.register_env("RadSearch-v1", lambda cfg: RadSearch(**cfg))
    result = train()