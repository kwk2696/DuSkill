import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerBoxCloseV2Policy(Policy):
    
    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lid_pos': obs[4:7],
            'extra_info_1': obs[7:-3],
            'box_pos': obs[-3:-1],
            'extra_info_2': obs[-1],
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
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.02])
        # pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([-.04, .0, .0]) # Mod: 1-change drift
        pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([.0, .0, .0]) 

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.02:
            targ_p = np.array([*pos_lid[:2], 0.2])
            return targ_p
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            targ_p = pos_lid
            return targ_p
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_box[2]) > 0.04:
            targ_p = np.array([pos_curr[0], pos_curr[1], pos_box[2]])
            return targ_p
        # Move to the goal
        else:
            targ_p = pos_box
            return targ_p

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_lid = o_d['lid_pos'] + np.array([.0, .0, +.02])

        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01 or abs(pos_curr[2] - pos_lid[2]) > 0.13:
            return .5
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 1.
