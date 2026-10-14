import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move, assert_fully_parsed
from metaworld.policies.policy import move_u, move_acc


class CustomSawyerReturnV2Policy(Policy):
    def __init__(self, env_cls: str = 'default', nfunc: float = None):
        self.second_objective=False
        self.env_cls = env_cls
        self.nfunc = nfunc
        self.step = 0

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'tcp_opened': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:],
        }

    def _desired_pos(o_d):
        pass

    def get_action(self, obs, obt = None, p=.5, prior_skill=None, prior_act=None):
        o_d = self._parse_obs(obs)
        pos_curr = o_d['hand_pos']

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })
        
        if self.nfunc is None:
            nfunc = p
        else:
            nfunc = self.nfunc
        
        if self.env_cls == 'default' or 'SP' in self.env_cls:
            if prior_skill == 'box' :
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.5 : 
                        action['delta_pos'] = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 0.1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'door' :
                if o_d['hand_pos'][2] < 0.19 :
                    action['delta_pos'] = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=nfunc)
                else:
                    action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif  prior_skill == 'stick':
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][2] < 0.2 : 
                        action['delta_pos'] = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'drawer':
                if o_d['hand_pos'][1] < 0.3 :
                    action['delta_pos'] = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([0, -1, 0.03]), p=nfunc)
                else :
                    action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
            elif prior_skill == 'puck':
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.4 : 
                        action['delta_pos'] = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'handle':
                action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
            else:
                action['delta_pos'] = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
        
        elif 'EN' in self.env_cls:
            target_vel = np.array([0, 0, 0])
            if prior_skill == 'box' :
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.5 : 
                        target_vel = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 0.1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'door' :
                if o_d['hand_pos'][2] < 0.25 :
                    target_vel = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=nfunc)
                else:
                    target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif  prior_skill == 'stick':
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][2] < 0.2 : 
                        target_vel = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'drawer':
                if o_d['hand_pos'][1] < 0.3 :
                    target_vel = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([0, -1, 0.03]), p=nfunc)
                else :
                    target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
            elif prior_skill == 'puck':
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.4 : 
                        target_vel = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 1]), p=nfunc)
                    else:
                        self.second_objective = True
                else :
                    target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
            elif prior_skill == 'handle':
                target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
            else:
                target_vel = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=nfunc)
                action['grab_effort'] = .5
            
            self.step += 1 
            temp = np.clip(0.1 * self.step, 0, 1)
            acc = move_acc(target_vel, obt[-3:]) * temp
            action['delta_pos'] = acc # obt[-3:] + acc * 0.1
        
        elif 'WD' in self.env_cls:
            delta_pos = np.array([0, 0, 0])
            if prior_skill == 'box' :
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.5 : 
                        delta_pos = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 0.1]), p=.425)
                    else:
                        self.second_objective = True
                else :
                    delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
            elif prior_skill == 'door' :
                if o_d['hand_pos'][2] < 0.19 :
                    delta_pos = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=.425)
                else:
                    delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
            elif  prior_skill == 'stick':
                if 0.25 <= o_d['tcp_opened'] <= 0.6:
                    action['grab_effort'] = -.65
                if self.second_objective is False:
                    if o_d['hand_pos'][2] < 0.2 : 
                        delta_pos = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([1, 0, 1]), p=.425)
                    else:
                        self.second_objective = True
                else :
                    delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
            elif prior_skill == 'drawer':
                if o_d['hand_pos'][1] < 0.3 :
                    delta_pos = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([0, -1, 0.03]), p=.425)
                else :
                    delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
                action['grab_effort'] = .5
            elif prior_skill == 'puck':
                if self.second_objective is False:
                    if o_d['hand_pos'][0] > 0.4 : 
                        delta_pos = move_u(o_d['hand_pos'], o_d['hand_pos'] + np.array([-1, 0, 1]), p=.425)
                    else:
                        self.second_objective = True
                else :
                    delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
            elif prior_skill == 'handle':
                delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
                action['grab_effort'] = .5
            else:
                delta_pos = move_u(o_d['hand_pos'], to_xyz=(0, 0.5, 0.2), p=.425)
                action['grab_effort'] = .5
           
            action['delta_pos'] = delta_pos #+ np.array([nfunc, nfunc, 0])

        return action.array
