import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerDrawerCloseV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_grasp_info': obs[3],
            'drwr_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if pos_curr[1] > pos_drwr[1]:
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
        else:
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
        action['grab_effort'] = 1.
    
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        # if further forward than the drawer...
        if pos_curr[1] > pos_drwr[1]:
            # if pos_curr[2] < pos_drwr[2] + 0.23:
            #     # rise up quickly (Z direction)
            #     return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5])
            # else:
            # move to front edge of drawer handle, but stay high in Z
            return pos_drwr + np.array([0., -0.075, 0.23])
        # drop down to touch drawer handle
        if np.linalg.norm(pos_curr - pos_drwr) > 0.12:
            return pos_drwr + np.array([0., -0.1, -0.0])
        # push toward drawer handle's centroid
        else:
            return pos_drwr


class CustomEnergySawyerDrawerCloseV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_grasp_info': obs[3],
            'drwr_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def reset(self):
        self.step = [0, 0, 0]
    
    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        
        desired_pos, mode = self._desired_pos(o_d)
        target_vel = move_u(o_d['hand_pos'], to_xyz=desired_pos, p=.5)
        action['grab_effort'] = 1.
        
        if mode == 2 and pos_curr[1] > 0.82:
            target_vel = np.array([0., 0., 0.])
        
        self.step[mode] += 1
        temp = np.clip(0.1 * self.step[mode], 0, 1)
        temp = 1 
        acc = move_acc(target_vel, obt[-3:]) * temp
        action['delta_pos'] = acc * nfunc # obt[-3:] + acc * 0.1
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])
        
        if pos_curr[1] > pos_drwr[1]:
            return pos_drwr + np.array([0., -0.075, 0.23]), 0
        if np.linalg.norm(pos_curr - pos_drwr) > 0.12:
            return pos_drwr + np.array([0., -0.1, -0.0]), 1
        else:
            return pos_drwr, 2


class CustomWindSawyerDrawerCloseV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'unused_grasp_info': obs[3],
            'drwr_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if pos_curr[1] > pos_drwr[1]:
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
        else:
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
        action['grab_effort'] = 1.
    
        action['delta_pos'] = delta_pos #+ np.array([nfunc, nfunc, 0])
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_drwr = o_d['drwr_pos'] + np.array([.0, .0, -.02])

        # if further forward than the drawer...
        if pos_curr[1] > pos_drwr[1]:
            # if pos_curr[2] < pos_drwr[2] + 0.23:
            #     # rise up quickly (Z direction)
            #     return np.array([pos_curr[0], pos_curr[1], pos_drwr[2] + 0.5])
            # else:
            # move to front edge of drawer handle, but stay high in Z
            return pos_drwr + np.array([0., -0.075, 0.23])
        # drop down to touch drawer handle
        if np.linalg.norm(pos_curr - pos_drwr) > 0.12:
            return pos_drwr + np.array([0., -0.1, -0.0])
        # push toward drawer handle's centroid
        else:
            return pos_drwr
