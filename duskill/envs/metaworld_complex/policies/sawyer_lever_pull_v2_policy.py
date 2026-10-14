import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerLeverPullV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lever_pos': obs[4:7],
            'unused_info': obs[7:-3],
            'goal_pos': obs[-3:],
        }

    def get_action(self, obs, obt=None, p=.5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc
            
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_lever = o_d['lever_pos'] + np.array([.0, -.055, .0])

        if np.linalg.norm(pos_curr[:2] - pos_lever[:2]) > 0.02:
            return pos_lever + np.array([0., 0., -0.1])
        elif abs(pos_curr[2] - pos_lever[2]) > 0.02:
            return pos_lever
        else:
            return o_d['goal_pos'] + np.array([.0, -.04, .0])
