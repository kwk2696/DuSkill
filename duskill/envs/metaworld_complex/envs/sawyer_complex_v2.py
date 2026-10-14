import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

INIT_CONFIG = {}

INIT_CONFIG['puck'] = \
{
    'puck_obj_init_pos': np.array([0.2, 0.5, 0.], dtype=np.float32),
    'puck_obj_init_angle': 0.3,
    'puck_goal_init_pos': np.array([0.5, 0.5, 0.], dtype=np.float32),
    'puck_goal_init_angle': np.array([-0.70738827, 0, 0, 0.70682518], dtype=np.float32),
    'puck_target_pos': np.array([0.47, 0.5, 0.02])
}

INIT_CONFIG['coffee'] = \
{
    'coffee_obj_init_angle': 0.3,
    'coffee_obj_init_pos': np.array([-0.25, 0.5, 0.]),
    'coffee_machine_init_pos': np.array([-0.55, 0.80, 0.], dtype=np.float32),
    'coffee_machine_init_angle': np.array([-0.95966165, 0, 0, -0.28115745], dtype=np.float32),
    'coffee_target_pos': np.array([-0.45, 0.65, 0.], dtype=np.float32),
}

INIT_CONFIG['door'] = \
{
    'door_obj_init_pos': np.array([-0.55, 0.75, 0.15], dtype=np.float32),
    'door_obj_init_angle': np.array([-0.95966165, 0, 0, -0.28115745], dtype=np.float32),
    'door_target_pos': np.array([-0.42, 0.25, 0.15], dtype=np.float32),
}

INIT_CONFIG['stick'] = \
{
    'stick_obj_init_pos': np.array([-0.2, 0.5, 0.], dtype=np.float32),
    'stick_obj_init_angle': 0.3,
    'stick_goal_init_pos': np.array([-0.6, 0.6, 0.], dtype=np.float32),
    'stick_goal_init_angle': np.array([-0.70738827, 0, 0, 0.70682518], dtype=np.float32),
    'stick_target_pos': np.array([-0.6, 0.6, 0.02])
}

INIT_CONFIG['box'] = \
{
    'box_obj_init_pos': np.array([0.25, 0.5, 0.], dtype=np.float32),
    'box_obj_init_angle': 0.3,
    'box_goal_init_pos': np.array([0.5, 0.6, 0.], dtype=np.float32),
    'box_target_pos': np.array([0.5, 0.55, 0.084])
}

INIT_CONFIG['handle'] = \
{
    'handle_obj_init_pos': np.array([0.27, 0.92, 0.0]),
}

INIT_CONFIG['button'] = \
{
    'button_obj_init_pos': np.array([-0.12, 0.95, 0.115]),
}

INIT_CONFIG['lever'] = \
{
    'lever_obj_init_pos': np.array([-0.15, 0.90, 0.0]),
}

INIT_CONFIG['drawer'] = \
{
    'drawer_obj_init_angle': np.array([0.3, ], dtype=np.float32),
    'drawer_obj_init_pos': np.array([0.2, 1.05, 0.]),
}


class SawyerComplexEnvV2(SawyerXYZEnv):
    LEVER_RADIUS = 0.2
    HANDLE_RADIUS = 0.02
    TARGET_RADIUS = 0.04
    THRESHOLD_COEF = 0.5
    
    TASK_THRESHOLD = {
            'lever': np.pi / 6,
            'coffee': 0.03,
            'button': 0.03,
            'puck': 0.07,
            'handle': 0.02,
            'stick': 0.07,
            'box': 0.08,
            'drawer': 0.05,
            'door': 0.08,
    }

    RELEASED_TASK_THRESHOLD = {
            'lever': np.pi / 6,
            'coffee': 0.03,
            'button': 0.04, # <- 0.02
            'puck': 0.12, # <- 0.1
            'handle': 0.02,
            'stick': 0.07,
            'box': 0.08,
            'drawer': 0.05,
            'door': 0.08,
    }
    ALL_SKILLS = ['box', 'puck', 'handle', 'drawer', 'button', 'lever', 'stick', 'door']

    def __init__(self, skill_list):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        lever_obj_low = (-0.1, 0.7, 0.0)
        lever_obj_high = (0.1, 0.8, 0.0)
        self.side_position = ['puck', 'coffee', 'stick']
        self.center_position = ['lever', 'handle', 'button']
        self.position = None

        self.mode = 0
        self.skill_list = skill_list
        self.skill_cnt = len(skill_list)
        super().__init__(
                self.model_name,
                hand_low=hand_low,
                hand_high=hand_high,
        )

        self._set_task_called = True
        self.init_config = {
                'coffee_obj_init_angle': 0.3,
                'coffee_obj_init_pos': np.array([0.3, 0.5, 0.]),
                'puck_obj_init_angle': 0.3,
                'puck_obj_init_pos': np.array([-0.2, 0.5, 0.], dtype=np.float32),
                'button_obj_init_pos': np.array([0.12, 0.9, 0.115], dtype=np.float32),
                'lever_obj_init_pos': np.array([-0.25, 0.9, 0.0], dtype=np.float32),
                'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
                'handle_init_pos': np.array([-0.25, 0.92, 0.0]),
        }

        self.obj_init_pos = self.init_config['puck_obj_init_pos']
        self.obj_init_angle = self.init_config['puck_obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.hand_init_pos_save = self.init_config['hand_init_pos'].copy()
        self.random_init = False

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
                np.array(lever_obj_low),
                np.array(lever_obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.15
        self.target_reward = 1000 * self.maxDist + 1000 * 2
        # peg insert side, handle,

    @property
    def model_name(self):
        pos = [['box', 'puck'], ['handle', 'drawer'], ['button', 'lever'], ['stick', 'door']]
        path_val = []
        for obj_list in pos:
            if obj_list[0] in self.skill_list:
                path_val.append(obj_list[0])
            elif obj_list[1] in self.skill_list:
                path_val.append(obj_list[1])
            else:
                raise ValueError
        return full_v2_path_for(
            'sawyer_xyz/sawyer_multitask_{}_{}_{}_{}.xml'.format(path_val[0], path_val[1], path_val[2], path_val[3]))

    def set_xyz_action(self, action, action_noize=None):
        if action_noize is not None:
            action += action_noize
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
                new_mocap_pos[0, :],
                self.mocap_low,
                self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _set_drawer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        self.set_state(qpos, qvel)

    def _set_door_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def evaluate_state(self, obs, action):
        (
            reward,
            _first_info,
            _second_info,
            _third_info,
            _fourth_info,
             _fifth_info
        ) = self.compute_reward(action, obs)

        meta_data = [_first_info, _second_info, _third_info, _fourth_info, _fifth_info]
        mode = self.skill_list[self.mode]
        
        info = {
            'success': float(_third_info <= self.RELEASED_TASK_THRESHOLD[mode] * self.THRESHOLD_COEF),
            'meta_data': meta_data
        }
        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self, mode=None):
        if mode is None:
            mode = self.skill_list[self.mode]
        if mode == 'lever':
            return self.lever_get_pos_objects()
        elif mode == 'coffee':
            return self.coffee_get_pos_objects()
        elif mode == 'button':
            return self.button_get_pos_objects()
        elif mode == 'puck':
            return self.puck_get_pos_objects()
        elif mode == 'handle':
            return self.handle_get_pos_objects()
        elif mode == 'stick':
            return self.stick_get_pos_objects()
        elif mode == 'box':
            return self.box_get_pos_objects()
        elif mode == 'drawer':
            return self.drawer_get_pos_objects()
        elif mode == 'door':
            return self.door_get_pos_objects()

    def _get_quat_objects(self, mode=None):
        if mode is None:
            mode = self.skill_list[self.mode]
        if mode == 'lever':
            return self.lever_get_quat_objects()
        elif mode == 'coffee':
            return self.coffee_get_quat_objects()
        elif mode == 'button':
            return self.button_get_quat_objects()
        elif mode == 'puck':
            return self.puck_get_quat_objects()
        elif mode == 'handle':
            return self.handle_get_quat_objects()
        elif mode == 'stick':
            return self.stick_get_quat_objects()
        elif mode == 'box':
            return self.box_get_quat_objects()
        elif mode == 'drawer':
            return self.drawer_get_quat_objects()
        elif mode == 'door':
            return self.door_get_quat_objects()

    def compute_reward(self, action, obs):
        mode = self.skill_list[self.mode]
        if mode == 'lever':
            return self.lever_compute_reward(action, obs)
        elif mode == 'coffee':
            return self.coffee_compute_reward(action, obs)
        elif mode == 'button':
            return self.button_compute_reward(action, obs)
        elif mode == 'puck':
            return self.puck_compute_reward(action, obs)
        elif mode == 'handle':
            return self.handle_compute_reward(action, obs)
        elif mode == 'stick':
            return self.stick_compute_reward(action, obs)
        elif mode == 'box':
            return self.box_compute_reward(action, obs)
        elif mode == 'drawer':
            return self.drawer_compute_reward(action, obs)
        elif mode == 'door':
            return self.door_compute_reward(action, obs)

    def set_position(self, position):
        self.position = position

    def reset_model(self):
        self.init_config = {'obj_init_pos': np.array([0., 0., 0.])}
        for obj in self.skill_list:
            self.init_config.update(INIT_CONFIG[obj])

        # hand custom init position
        self.hand_init_pos = self.hand_init_pos_save.copy() + np.random.uniform(-1, 1, (3)) * np.array([0.1, 0.05, 0.05])  # uniform * scale
        self._reset_hand()
        self.mode = 0
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        if 'coffee' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('coffee_machine')] = self.init_config['coffee_machine_init_pos']
            self.sim.model.body_quat[self.model.body_name2id('coffee_machine')] = self.init_config['coffee_machine_init_angle']
            self._coffee_target_pos = self.init_config['coffee_target_pos']
            self.sim.model.body_pos[self.model.body_name2id('coffee_mug')] = self.init_config['coffee_obj_init_pos']
        if 'puck' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('puck_goal')] = self.init_config['puck_goal_init_pos']
            self.sim.model.body_quat[self.model.body_name2id('puck_goal')] = self.init_config['puck_goal_init_angle']
            self.sim.model.body_pos[self.model.body_name2id('puck_channel')] = self.init_config['puck_obj_init_pos']
            self._puck_target_pos = self.init_config['puck_target_pos']
        if 'button' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('btnbox')] = self.init_config['button_obj_init_pos']
            self._button_target_pos = self.init_config['button_obj_init_pos'] + np.array([0., -0.115, 0.])
            self._button_obj_to_target_init = abs(self._button_target_pos[1] - self._get_site_pos('buttonStart')[1])
        if 'lever' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('lever')] = self.init_config['lever_obj_init_pos']
            self._lever_pos_init = self.init_config['lever_obj_init_pos'] + np.array([.12, -self.LEVER_RADIUS, .25])
            self._lever_target_pos = self.init_config['lever_obj_init_pos'] + np.array([.12, .0, .25 + self.LEVER_RADIUS])
        if 'handle' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('handlebox')] = self.init_config['handle_obj_init_pos']
            self._handle_target_pos = self._get_site_pos('goalPress')
            self._handle_init_pos = self._get_pos_objects('handle')
        if 'stick' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('peg')] = self.init_config['stick_obj_init_pos']
            self.sim.model.body_pos[self.model.body_name2id('stickbox')] = self.init_config['stick_goal_init_pos']
            self.sim.model.body_quat[self.model.body_name2id('stickbox')] = self.init_config['stick_goal_init_angle']
            if self.init_config['stick_goal_init_pos'][0] < 0:
                self._stick_target_pos = self.init_config['stick_goal_init_pos'] + np.array([.03, .0, .13])
                self.peg_head_pos_init = self._get_site_pos('pegHead')
            else:
                self._stick_target_pos = self.init_config['stick_goal_init_pos'] + np.array([-.03, .0, .13])
                self.peg_head_pos_init = self._get_site_pos('pegEnd')
        if 'box' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('boxbody')] = self.init_config['box_goal_init_pos']
            self._box_target_pos = self.init_config['box_target_pos']
            self.sim.model.body_pos[self.model.body_name2id('boxbodytop')] = self.init_config['box_obj_init_pos']
        if 'drawer' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('drawer')] = self.init_config['drawer_obj_init_pos']
            self._drawer_init_pos = self._get_pos_objects('drawer')
            self._drawer_target_pos = self.init_config['drawer_obj_init_pos'] + np.array([.0, -.16, .14])
            self._set_drawer_xyz(-0.15)
        if 'door' in self.skill_list:
            self.sim.model.body_pos[self.model.body_name2id('door')] = self.init_config['door_obj_init_pos']
            self._door_init_pos = self._get_pos_objects('door')
            self._door_target_pos = self.init_config['door_target_pos']

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init else self.init_config['obj_init_pos']
        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos_list = []
        for skill in self.skill_list:
            if skill == 'button':
                self._target_pos_list.append(self._button_target_pos)
            elif skill == 'coffee':
                self._target_pos_list.append(self._coffee_target_pos)
            elif skill == 'puck':
                self._target_pos_list.append(self._puck_target_pos)
            elif skill == 'lever':
                self._target_pos_list.append(self._lever_target_pos)
            elif skill == "handle":
                self._target_pos_list.append(self._handle_target_pos)
            elif skill == "stick":
                self._target_pos_list.append(self._stick_target_pos)
            elif skill == 'box':
                self._target_pos_list.append(self._box_target_pos)
            elif skill == 'drawer':
                self._target_pos_list.append(self._drawer_target_pos)
            elif skill == 'door':
                self._target_pos_list.append(self._door_target_pos)
        self._target_pos = self._target_pos_list[self.mode]

        self._last_stable_obs = self._get_obs()
        return self.get_all_observations()

    @_assert_task_is_set
    def step(self, action, action_noize=None):

        if action_noize is not None:
            self.set_xyz_action(action[:3], action_noize[:3])
        else:
            self.set_xyz_action(action[:3])

        self.do_simulation([action[-1], -action[-1]])
        self.curr_path_length += 1

        if self._did_see_sim_exception:
            return (
                self.get_all_observations(),  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,  # termination flag always False
                {  # info
                    'success': False,
                    'near_object': 0.0,
                    'grasp_success': False,
                    'grasp_reward': 0.0,
                    'in_place_reward': 0.0,
                    'obj_to_target': 0.0,
                    'unscaled_reward': 0.0,
                }
            )

        self._last_stable_obs = self._get_obs()

        if not self.isV2:
            # v1 environments expect this superclass step() to return only the
            # most recent observation. they override the rest of the
            # functionality and end up returning the same sort of tuple that
            # this does
            return self.get_all_observations()

        reward, info = self.evaluate_state(self._last_stable_obs, action)
        done = False
        info['skill'] = self.skill_list[self.mode]
        info['skill_seq'] = self.skill_list
        info['video_info'] = {
            'qpos': self.data.qpos.flat.copy(),
            'qvel': self.data.qvel.flat.copy()
        }

        dict_obs = self.get_all_dict_observations()

        if info['success']:
            reward = 1
            self.mode += 1
            if self.mode == self.skill_cnt:
                done = True
            else:
                if len(self.skill_list) == 1:  # for single skill video collection mode is always 0
                    self.mode = 0
                self._target_pos = self._target_pos_list[self.mode]
        else:
            reward = 0
        return self.get_all_observations(dict_obs), reward, done, info

    def _get_obs(self):
        # do frame stacking
        mode = self.skill_list[self.mode]
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        if self.isV2:
            obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        else:
            obs = np.hstack((curr_obs, pos_goal))
        self._prev_obs = curr_obs
        return obs

    def _get_pos_goal(self, mode=None):
        """
        Retrieves goal position from mujoco properties or instance vars
        Returns: np.ndarray: Flat array (3 elements) representing the goal position
        """
        if mode is None:
            mode = self.skill_list[self.mode]
        if mode == 'lever':
            return self._lever_target_pos
        elif mode == 'coffee':
            return self._coffee_target_pos
        elif mode == 'button':
            return self._button_target_pos
        elif mode == 'puck':
            return self._puck_target_pos
        elif mode == 'handle':
            return self._handle_target_pos
        elif mode == 'stick':
            return self._stick_target_pos
        elif mode == 'box':
            return self._box_target_pos
        elif mode == 'drawer':
            return self._drawer_target_pos
        elif mode == 'door':
            return self._door_target_pos

    def _get_curr_obs_combined_no_goal(self):
        """
        Combines the end effector's {pos, closed amount} and the object(s)'
        {pos, quat} into a single flat observation. The goal's position is
        *not* included in this.

        Returns: np.ndarray: The flat observation array (18 elements)
        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0

        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        if self.isV2:
            obj_quat = self._get_quat_objects()
            assert len(obj_quat) % 4 == 0
            obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
            obs_obj_padded[:len(obj_pos) + len(obj_quat)] = np.hstack([
                    np.hstack((pos, quat))
                    for pos, quat in zip(obj_pos_split, obj_quat_split)
            ])
            assert (len(obs_obj_padded) in self._obs_obj_possible_lens)
            return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
        else:
            # is a v1 environment
            obs_obj_padded[:len(obj_pos)] = obj_pos
            assert (len(obs_obj_padded) in self._obs_obj_possible_lens)
            return np.hstack((pos_hand, obs_obj_padded))

    def get_all_dict_observations(self):
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )

        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        observation = {}
        observation['robot_obs'] = np.hstack([pos_hand, gripper_distance_apart])

        for skill in self.ALL_SKILLS:
            if skill in self.skill_list:
                obs_obj_padded = np.zeros(self._obs_obj_max_len)

                obj_pos = self._get_pos_objects(mode=skill)
                assert len(obj_pos) % 3 == 0

                obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

                if self.isV2:
                    obj_quat = self._get_quat_objects(mode=skill)
                    assert len(obj_quat) % 4 == 0
                    obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
                    obs_obj_padded[:len(obj_pos) + len(obj_quat)] = np.hstack([
                        np.hstack((pos, quat))
                        for pos, quat in zip(obj_pos_split, obj_quat_split)
                    ])
                    assert (len(obs_obj_padded) in self._obs_obj_possible_lens)
                observation[skill] = np.hstack([obs_obj_padded, self._get_pos_goal(mode=skill)])
            else:
                observation[skill] = np.zeros(self._obs_obj_max_len + 3)
        return observation

    def get_all_observations(self, dict_obs=None):
        if dict_obs is None:
            dict_obs = self.get_all_dict_observations()
        obs = dict_obs['robot_obs']
        for skill in self.ALL_SKILLS:
            obs = np.hstack([obs, dict_obs[skill]])
        return obs

    def door_get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def door_get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('handle')).as_quat()

    def lever_get_pos_objects(self):
        return self._get_site_pos('leverStart')

    def lever_get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('objGeom')).as_quat()

    def button_get_pos_objects(self):
        return self.get_body_com('button') + np.array([.0, -.193, .0])

    def button_get_quat_objects(self):
        return self.sim.data.get_body_xquat('button')

    def coffee_get_pos_objects(self):
        return self.get_body_com('coffee_mug')

    def coffee_get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('mug')
        ).as_quat()

    def handle_get_pos_objects(self):
        return self._get_site_pos('handleStart')

    def handle_get_quat_objects(self):
        return np.zeros(4)

    def puck_get_pos_objects(self):
        return self.data.get_geom_xpos('puck') + np.array([0., 0., 0.05])

    def puck_get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('puck')).as_quat()

    def box_get_pos_objects(self):
        return self.get_body_com('top_link')

    def box_get_quat_objects(self):
        return self.sim.data.get_body_xquat('top_link')

    def stick_get_pos_objects(self):
        return self._get_site_pos('pegGrasp')

    def stick_get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_site_xmat('pegGrasp')).as_quat()

    def drawer_get_pos_objects(self):
        return self.get_body_com('drawer_link') + np.array([.0, -.16, .05])

    def drawer_get_quat_objects(self):
        return np.zeros(4)

    def lever_compute_reward(self, action, obs):
        gripper = obs[:3]
        lever = obs[4:7]

        # De-emphasize y error so that we get Sawyer's shoulder underneath the
        # lever prior to bumping on against
        scale = np.array([4., 1., 4.])
        # Offset so that we get the Sawyer's shoulder underneath the lever,
        # rather than its fingers
        offset = np.array([.0, .055, .07])

        shoulder_to_lever = (gripper + offset - lever) * scale
        shoulder_to_lever_init = (self.init_tcp + offset - self._lever_pos_init) * scale

        # This `ready_to_lift` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually lifting the lever
        ready_to_lift = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lever),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lever_init),
            sigmoid='long_tail',
        )

        # The skill of the agent should be measured by its ability to get the
        # lever to point straight upward. This means we'll be measuring the
        # current angle of the lever's joint, and comparing with 90deg.
        lever_angle = -self.data.get_joint_qpos('LeverAxis')
        lever_angle_desired = np.pi / 2.0

        lever_error = abs(lever_angle - lever_angle_desired)

        # We'll set the margin to 15deg from horizontal. Angles below that will
        # receive some reward to incentivize exploration, but we don't want to
        # reward accidents too much. Past 15deg is probably intentional movement
        lever_engagement = reward_utils.tolerance(
            lever_error,
            bounds=(0, np.pi / 48.0),
            margin=(np.pi / 2.0) - (np.pi / 12.0),
            sigmoid='long_tail'
        )

        target = self._lever_target_pos
        obj_to_target = np.linalg.norm(lever - target)
        in_place_margin = (np.linalg.norm(self._lever_pos_init - target))

        in_place = reward_utils.tolerance(obj_to_target, bounds=(0, 0.04), margin=in_place_margin, sigmoid='long_tail', )

        reward = 10.0 * reward_utils.hamacher_product(ready_to_lift, in_place)
        return (
            reward,
            np.linalg.norm(shoulder_to_lever),
            ready_to_lift,
            lever_error,
            lever_engagement,
            None
        )

    def button_compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._button_target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self._button_obj_to_target_init,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )

    def coffee_compute_reward(self, action, obs):
        obj = obs[4:7]
        target = self._coffee_target_pos.copy()

        # Emphasize X and Y errors
        scale = np.array([2., 2., 1.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.init_config['coffee_obj_init_pos'] - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, 0.05),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.04,
            obj_radius=0.02,
            pad_success_thresh=0.05,
            xz_thresh=0.05,
            desired_gripper_effort=0.7,
            medium_density=True
        )

        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.04 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < 0.05:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - target),  # recompute to avoid `scale` above
            object_grasped,
            in_place
        )

    def puck_compute_reward(self, actions, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._puck_target_pos

        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.init_config['puck_obj_init_pos'] - target)
        in_place = reward_utils.tolerance(obj_to_target, bounds=(
            0, 
            _TARGET_RADIUS), 
            margin=in_place_margin - _TARGET_RADIUS,
            sigmoid='long_tail', 
        )

        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(self.init_tcp - self.init_config['puck_obj_init_pos'])
        object_grasped = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, _TARGET_RADIUS),
            margin=obj_grasped_margin - _TARGET_RADIUS,
            sigmoid='long_tail',
        )

        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped,
            in_place
        )
        reward = 1.5 * object_grasped

        if tcp[2] <= 0.03 and tcp_to_obj < 0.07:
            reward = 2 + (7 * in_place)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        )

    def handle_compute_reward(self, actions, obs):
        del actions

        objPos = obs[4:7]
        obj = self._get_pos_objects()
        tcp = self.tcp_center
        target = self._handle_target_pos.copy()

        target_to_obj = (obj[2] - target[2])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self._handle_init_pos[2] - target[2])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.HANDLE_RADIUS),
            margin=abs(target_to_obj_init - self.HANDLE_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self._handle_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = 1 if target_to_obj <= self.HANDLE_RADIUS else reward
        reward *= 10
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )

    def stick_compute_reward(self, action, obs):
        tcp = self.tcp_center
        obj = obs[4:7]
        if self.init_config['stick_goal_init_pos'][0] < 0:
            obj_head = self._get_site_pos('pegHead')
        else:
            obj_head = self._get_site_pos('pegEnd')
        tcp_opened = obs[3]
        target = self._stick_target_pos
        tcp_to_obj = np.linalg.norm(obj - tcp)
        scale = np.array([1., 2., 2.])
        #  force agent to pick up object then insert
        obj_to_target = np.linalg.norm((obj_head - target) * scale)

        in_place_margin = np.linalg.norm((self.peg_head_pos_init - target) * scale)
        in_place = reward_utils.tolerance(obj_to_target,
                                                                          bounds=(0, self.TARGET_RADIUS),
                                                                          margin=in_place_margin,
                                                                          sigmoid='long_tail', )
        ip_orig = in_place
        brc_col_box_1 = self._get_site_pos('bottom_right_corner_collision_box_1')
        tlc_col_box_1 = self._get_site_pos('top_left_corner_collision_box_1')

        brc_col_box_2 = self._get_site_pos('bottom_right_corner_collision_box_2')
        tlc_col_box_2 = self._get_site_pos('top_left_corner_collision_box_2')
        collision_box_bottom_1 = reward_utils.rect_prism_tolerance(
            curr=obj_head,
            one=tlc_col_box_1,
            zero=brc_col_box_1
        )
        collision_box_bottom_2 = reward_utils.rect_prism_tolerance(
            curr=obj_head,
            one=tlc_col_box_2,
            zero=brc_col_box_2
        )
        collision_boxes = reward_utils.hamacher_product(
            collision_box_bottom_2,
            collision_box_bottom_1
        )
        in_place = reward_utils.hamacher_product(
            in_place,
            collision_boxes
        )

        pad_success_margin = 0.03
        object_reach_radius = 0.01
        x_z_margin = 0.005
        obj_radius = 0.0075

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=object_reach_radius,
            obj_radius=obj_radius,
            pad_success_thresh=pad_success_margin,
            xz_thresh=x_z_margin,
            high_density=True
        )
        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.init_config['stick_obj_init_pos'][2]):
            object_grasped = 1.
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped,
            in_place
        )
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.init_config['stick_obj_init_pos'][2]):
            reward += 1. + 5 * in_place

        if obj_to_target <= 0.07:
            reward = 10.

        return reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place

    def box_compute_reward(self, actions, obs):
        reward_grab = (np.clip(actions[3], -1, 1) + 1.0) / 2.0
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        reward_quat = max(1.0 - error / 0.2, 0.0)

        target_pos = self._box_target_pos.copy()

        hand = obs[:3]
        lid = obs[4:7] + np.array([.0, .0, .02])

        threshold = 0.02
        # floor is a 3D funnel centered on the lid's handle
        radius = np.linalg.norm(hand[:2] - lid[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # grab the lid's handle
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - lid),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor, in_place)

        # now actually put the lid on the box
        pos_error = target_pos - lid
        error_scale = np.array([1., 1., 3.])  # Emphasize Z error
        a = 0.2  # Relative importance of just *trying* to lift the lid at all
        b = 0.8  # Relative importance of placing the lid on the box
        lifted = a * float(lid[2] > 0.04) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * error_scale),
            bounds=(0, 0.05),
            margin=0.25,
            sigmoid='long_tail',
        )

        reward_steps = ready_to_lift, lifted

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success
        success = np.linalg.norm(obs[4:7] - self._target_pos) < 0.08
        if success:
            reward = 10.0

        # STRONG emphasis on proper lid orientation to prevent reward hacking
        # (otherwise agent learns to kick-flip the lid onto the box)
        reward *= reward_quat

        return (
            reward,
            reward_grab,
            ready_to_lift,
            np.linalg.norm(obs[4:7] - self._target_pos),
            lifted,
            None
        )

    def drawer_compute_reward(self, action, obs):
        obj = obs[4:7]

        tcp = self.tcp_center
        target = self._drawer_target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self._drawer_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_reach_radius = 0.005
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self._drawer_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_reach_radius),
            margin=abs(tcp_to_obj_init - handle_reach_radius),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, action[-1]), 1)

        reach = reward_utils.hamacher_product(reach, gripper_closed)
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        if target_to_obj <= self.TARGET_RADIUS + 0.015:
            reward = 1.

        reward *= 10

        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )

    def door_compute_reward(self, actions, obs):
        theta = self.data.get_joint_qpos('doorjoint')

        reward_grab = (np.clip(actions[3], -1, 1) + 1.0) / 2.0
        hand = obs[:3]
        door = obs[4:7] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # move the hand to a position between the handle and the main door body
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.2  # Relative importance of just *trying* to open the door at all
        b = 0.8  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi / 90.) + b * reward_utils.tolerance(
            np.pi / 2. + np.pi / 6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi / 3.,
            sigmoid='long_tail',
        )

        reward = sum((
            2.0 * reward_utils.hamacher_product(ready_to_open, reward_grab),
            8.0 * opened,
        ))
        # Override reward on success flag
        if np.linalg.norm(obs[4:7] - self._door_target_pos) <= 0.08:
            reward = 10.0

        return (
            reward,
            reward_grab,
            ready_to_open,
            np.linalg.norm(obs[4:7] - self._door_target_pos),
            opened,
            None
        )
