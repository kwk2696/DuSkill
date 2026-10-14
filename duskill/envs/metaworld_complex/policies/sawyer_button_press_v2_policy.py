import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move
from metaworld.policies.policy import move_u, move_acc


class CustomSpeedSawyerButtonPressV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc
    
    @staticmethod
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt=None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button

        if not np.all(np.isclose(np.array([hand_x, hand_z]), np.array([button_initial_x, button_initial_z]), atol=0.05)):
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=nfunc) # go to the button
        else:
            action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=nfunc)
        action['grab_effort'] = 0.

        return action.array

    @staticmethod
    def desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button
        if not np.all(
            np.isclose(
                np.array([hand_x, hand_z]), np.array([button_initial_x, button_initial_z]),
                atol=0.05)):
            pos_button[1] = pos_curr[1] - .1
            return pos_button
        # if the hand is aligned with the button, push the button in, by
        # increasing the hand's y position
        pos_button[1] += 0.02
        return pos_button


class CustomEnergySawyerButtonPressV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc
    
    @staticmethod
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def reset(self):
        self.step = [0, 0]

    def get_action(self, obs, obt = None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button

        desired_pos, mode = self._desired_pos(o_d)
        target_vel = move_u(o_d['hand_pos'], to_xyz=desired_pos, p=.5) # go to the button
        action['grab_effort'] = 0.

        if mode == 1 and pos_curr[1] > 0.74:
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
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button
        if not np.all(
            np.isclose(
                np.array([hand_x, hand_z]), np.array([button_initial_x, button_initial_z]),
                atol=0.05)):
            pos_button[1] = pos_curr[1] - .1
            return pos_button, 0
        # if the hand is aligned with the button, push the button in, by
        # increasing the hand's y position
        pos_button[1] += 0.02
        return pos_button, 1


class CustomWindSawyerButtonPressV2Policy(Policy):

    def __init__(self, nfunc: float = None):
        self.nfunc = nfunc
    
    @staticmethod
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs, obt=None, p = .5):
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc

        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button

        if not np.all(np.isclose(np.array([hand_x, hand_z]), np.array([button_initial_x, button_initial_z]), atol=0.05)):
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=.425) # go to the button
        else:
            delta_pos = move_u(o_d['hand_pos'], to_xyz=self.desired_pos(o_d), p=.425)
        action['grab_effort'] = 0.
        
        action['delta_pos'] = delta_pos #+ np.array([nfunc, nfunc, 0])
        return action.array

    @staticmethod
    def desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_button = o_d['button_pos'] + np.array([0., 0., -0.02])

        # align the gripper with the button if the gripper does not have
        # the same x and z position as the button.
        hand_x, hand_y, hand_z = pos_curr
        button_initial_x, button_initial_y, button_initial_z = pos_button
        if not np.all(
            np.isclose(
                np.array([hand_x, hand_z]), np.array([button_initial_x, button_initial_z]),
                atol=0.05)):
            pos_button[1] = pos_curr[1] - .1
            return pos_button
        # if the hand is aligned with the button, push the button in, by
        # increasing the hand's y position
        pos_button[1] += 0.02
        return pos_button
