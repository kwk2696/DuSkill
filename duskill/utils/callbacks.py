import gym

from stable_baselines3.common.callbacks import BaseCallback
from duskill.utils.evaluations import evaluate_skill_policy


class EvalCallback(BaseCallback):
    """Periodically roll out the current policy and append the results to `eval_path`."""
    def __init__(self, env: gym.Env, eval_path: str, eval_freq: int, eval_episodes: int):
        super(EvalCallback, self).__init__(verbose=True)
        self.env = env
        self.eval_path = eval_path
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes

    def _init_callback(self) -> None:
        return

    def _on_step(self) -> None:
        if (self.n_calls == 1 or self.n_calls % self.eval_freq == 0):
            print(f"Evaluating at {self.n_calls}")
            evaluate_skill_policy(
                self.model,
                self.env,
                eval_episodes=self.eval_episodes,
                deterministic=True,
                ret_success=True,
                ret_std=True,
                save_path=self.eval_path+'/eval.txt',
            )
