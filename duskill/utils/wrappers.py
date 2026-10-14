import gym


class SkillWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, h: int = 1):
        super().__init__(env)
        self.h = h # skill length

    def step(self, actions, render=False):
        # Note that action is [action_size * h] length sequence
        actions = actions.reshape(self.h, -1)
        
        rews = 0.
        images = []
        for i, action in enumerate(actions):
            obs, rew, done, info = self.env.step(action)
            if render: images.append(self.env.render())
            rews += rew
            if done: break
        
        info['images'] = images
        return obs, rews, done, info # last info

