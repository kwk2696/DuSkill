import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerPlateSlideSideV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:36],
            'target_pos': obs[36:],
        }

    def get_action(self, obs, obt = None, p=.5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.03, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([+.08, -.03, -.04])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:  # Move toward puck
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
            action['grab_effort'] = 1.
        else:  # Push puck
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
            action['grab_effort'] = 0.
        
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([.08, -.03, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.03, -.04])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:
            return pos_puck + np.array([.0, 0, .05])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        else:
            return o_d['target_pos']


class CustomEnergySawyerPlateSlideSideV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc
        self.flag = False

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:36],
            'target_pos': obs[36:],
        }
    
    def reset(self):
        self.step = [0, 0, 0]

    def get_action(self, obs, obt = None, p=.5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.03, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([+.08, -.03, -.04])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        
        desired_pos, mode = self._desired_pos(o_d)
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:  # Move toward puck
            target_vel = move_u(o_d['hand_pos'], to_xyz=desired_pos, p=.5)
            action['grab_effort'] = 1.
        else:  # Push puck
            target_vel = move_u(o_d['hand_pos'], to_xyz=desired_pos, p=nfunc)
            action['grab_effort'] = 0.
        

        self.step[mode] += 1
        temp = np.clip(0.1 * self.step[mode], 0, 1)
        temp = 1 
        acc = move_acc(target_vel, obt[-3:]) * temp 
        action['delta_pos'] = acc * nfunc # obt[-3:] + acc * 0.1
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([.08, -.01, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.01, -.04])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:
            return pos_puck + np.array([.0, .0, .05]), 0
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck, 1
        else:
            return o_d['target_pos'], 2 


class CustomWindSawyerPlateSlideSideV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:36],
            'target_pos': obs[36:],
        }

    def get_action(self, obs, obt = None, p=.5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.03, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([+.08, -.03, -.04])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:  # Move toward puck
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
            action['grab_effort'] = 1.
        else:  # Push puck
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
            action['grab_effort'] = 0.

        action['delta_pos'] = delta_pos #+ np.array([nfunc, nfunc, 0])
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        if o_d['target_pos'][0] - pos_curr[0] < 0:
            pos_puck = o_d['puck_pos'] + np.array([.08, -.03, -.04])
        else:
            pos_puck = o_d['puck_pos'] + np.array([-.08, -.03, -.04])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.13:
            return pos_puck + np.array([.0, 0, .05])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        else:
            return o_d['target_pos']
