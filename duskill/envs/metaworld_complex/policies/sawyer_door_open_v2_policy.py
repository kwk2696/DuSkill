import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerDoorOpenV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'door_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)

        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:	# Move to door
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
        else: # Open the door
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=nfunc)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        # align end effector's Z axis with door handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:
            return pos_door + np.array([0.06, 0.02, 0.2])
        # drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([0.06, 0.02, 0.])
        # push from front edge toward door handle's centroid
        else:
            return pos_door


class CustomEnergySawyerDoorOpenV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'door_pos': obs[4:7],
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
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        
        desired_pos, mode = self._desired_pos(o_d)
        target_vel = move_u(o_d['hand_pos'], to_xyz=desired_pos, p=.5)
        action['grab_effort'] = 1.
        
        if mode == 2 and pos_curr[0] < -0.4:
            target_vel = np.array([0., 0., 0.])

        self.step[mode] += 1
        temp = np.clip(0.1 * self.step[mode], 0, 1)
        temp = 1
        acc = move_acc(target_vel, obt[-3:]) * temp
        action['delta_pos'] = acc * nfunc #obt[-3:] + acc * 0.1
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        # align end effector's Z axis with door handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:
            return pos_door + np.array([0.08, 0.016, 0.2]), 0
        # drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([0.06, 0.02, 0.]), 1
        # push from front edge toward door handle's centroid
        else:
            return pos_door, 2


class CustomWindSawyerDoorOpenV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'door_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)

        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:	# Move to door
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
        else: # Open the door
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=.425)
        action['grab_effort'] = 1.

        action['delta_pos'] = delta_pos #+ np.array([nfunc, nfunc, 0])
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_door = o_d['door_pos']
        pos_door[0] -= 0.05

        # align end effector's Z axis with door handle's Z axis
        if np.linalg.norm(pos_curr[:2] - pos_door[:2]) > 0.12:
            return pos_door + np.array([0.06, 0.02, 0.2])
        # drop down on front edge of door handle
        elif abs(pos_curr[2] - pos_door[2]) > 0.04:
            return pos_door + np.array([0.06, 0.02, 0.])
        # push from front edge toward door handle's centroid
        else:
            return pos_door
