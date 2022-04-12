from datetime import datetime
from src.teacher.dangerous_grid_world.dgw_single_switch import evaluate_single_switch_policy, SingleSwitchPolicy
import gym
from itertools import product
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
import tensorflow as tf
import numpy as np
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from src.students import LagrangianStudent, identity_transfer
from src.online_learning import ExponetiatedGradient
from src.envs import CMDP, DangerousGridWorldEnvCustom
from src.teacher import create_intervention
from src.envs.dangerous_grid_world.utils import get_intervention_states, create_intervention_from_list
from src.teacher.evaluation_loggers import BaseEvaluationLogger
from src.teacher.common import TeacherEnv
import os
import GPy
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def constraint(info=None, **kwargs):
    return {'g': float(info['next_state_type'] == 'danger')}


def base_cenv_fn():
    # get the base constrained env
    return CMDP(DangerousGridWorldEnvCustom(timeout=200), constraint,
                constraints_values=[0],
                n_constraints=1,
                avg_constraint=True)


domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-0.5, 5.5)},
              {'name': 'var_2', 'type': 'continuous', 'domain': (0, 0.2)},
              {'name': 'var_3', 'type': 'continuous', 'domain': (-0.5, 5.5)},
              {'name': 'var_4', 'type': 'continuous', 'domain': (0, 0.2)},
              {'name': 'var_5', 'type': 'discrete', 'domain': tuple(range(4))},
              {'name': 'var_6', 'type': 'discrete', 'domain': tuple(range(4))},
              {'name': 'var_7', 'type': 'discrete', 'domain': tuple(range(4))}]


def make_base_cenvs():
    # make intervention base constrained envs
    # dist =           [1,    1,    1,    1,]
    # tau =            [0.1,  0.1,  0,    0,]
    # buff_size =      [1,    0,    1,    0,]
    # avg_constraint = [True, True, True, True]
    interventions = []

    for d, t, b, avg in product([1], [0, 0.1], [1, 0], [True]):
        interventions.append(
            create_intervention(
                base_cenv_fn,
                create_intervention_from_list(get_intervention_states(d)),
                [t], b, use_vec=True, avg_constraint=avg)
        )

    assert callable(interventions[0])  # assert that the intervention is a function
    test_env = create_intervention(  # test when no intervention is present
        base_cenv_fn(),
        create_intervention_from_list(get_intervention_states()),
        [0.0], 0, avg_constraint=True)

    return interventions, test_env


def cnn_feature_extractor(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, **kwargs))
    layer_3 = conv_to_fc(layer_2)
    return activ(linear(layer_3, 'fc1', n_hidden=32, init_scale=np.sqrt(2)))


def create_teacher_env(new_br_kwargs={}, new_online_kwargs={},
                       original=False, obs_from_training=False,
                       non_stationary_bandit=False):
    # Student definition ==================================

    ## Best response algorithm parameters
    br_kwargs = dict(policy=CnnPolicy, verbose=0, n_steps=128,
                     ent_coef=0.05, cliprange=0.2, learning_rate=1e-3,
                     noptepochs=9,
                     policy_kwargs={'cnn_extractor': cnn_feature_extractor})
    br_kwargs.update(new_br_kwargs)

    ## Online algorithm parameters
    online_kwargs = dict(B=0.5, eta=1.0)
    online_kwargs.update(new_online_kwargs)

    student_default_kwargs = {
        'env': None,
        'br_algo': PPO2,
        'online_algo': ExponetiatedGradient,
        'br_kwargs': br_kwargs,
        'online_kwargs': online_kwargs,
        'lagrangian_ronuds': 2,
        'curriculum_transfer': identity_transfer,
        'br_uses_vec_env': True,
        'use_sub_proc_env': False,
        'n_envs': 4,
    }
    student_ranges_dict = {}

    # Teacher interventions ===============================

    if original:
        # To preserve the teacher env interface while training in the
        # original environment, we introduce a dummy intervention
        # condition that is always False.
        def dummy_intervention(**kwargs):
            return 0
        _, test_env = make_base_cenvs()
        intervention = create_intervention(
            base_cenv_fn,
            dummy_intervention,
            [0], 0, use_vec=True, avg_constraint=True)
        interventions = [intervention]
    else:
        interventions, test_env = make_base_cenvs()

    # todo: get clarity on these values
    learning_steps = 10000 * 2
    time_steps_lim = learning_steps * 10
    test_episode_timeout = 200
    test_episode_number = 10

    # todo: keep same for now
    if obs_from_training:
        env_cls = DangerousGridWorldTrainingObservation
    elif non_stationary_bandit:
        env_cls = TeacherEnv
    else:
        env_cls = DangerousGridWorldTeacherEnv

    return env_cls(student_cls=LagrangianStudent,
                   student_default_kwargs=student_default_kwargs,
                   interventions=interventions,
                   final_env=test_env,
                   logger_cls=BaseEvaluationLogger,
                   student_ranges_dict=student_ranges_dict,
                   learning_steps=learning_steps,
                   test_episode_number=test_episode_number,
                   test_episode_timeout=test_episode_timeout,
                   time_steps_lim=time_steps_lim,
                   normalize_obs=False)


class DangerousGridWorldEvaluationLogger(BaseEvaluationLogger):
    @staticmethod
    def determine_termination_cause(transition_dict):
        """Return -1 for failure, +1 for success and 0 for timeout"""
        if not transition_dict['done']:
            return None
        else:
            if transition_dict['info']['next_state_type'] == 'terminal':
                return 1
            elif transition_dict['info']['teacher_intervention']:
                return -1
            else:
                return 0


class DangerousGridWorldTeacherEnv(TeacherEnv):
    def __init__(self, student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):
        super().__init__(student_cls, student_default_kwargs, interventions,
                         final_env, logger_cls, student_ranges_dict,
                         learning_steps, test_episode_number,
                         test_episode_timeout, normalize_obs,
                         time_steps_lim, rounds_lim, cost_lim)
        if self.normalize_obs:
            raise NotImplementedError

        self.observation_space = gym.spaces.Box(low=np.inf, high=-np.inf,
                                                shape=(2,),
                                                dtype=np.float)
        self.steps_teacher_episode = 0
        self.steps_per_episode = int(time_steps_lim / learning_steps)

        # Counters for training failures
        self.student_failures = 0
        self.student_training_episodes_current_env = 0

    def reset(self):
        self.steps_teacher_episode = 0
        obs = super().reset()
        # Count training failures
        self.student_failures = 0
        self.student_training_episodes_current_env = 0
        self.set_student_training_logging(True)
        return obs

    def step(self, action):
        self.steps_teacher_episode += 1
        if action != self.old_action:
            self.student_training_episodes_current_env = 0
        obs, r, done, info = super().step(action)
        # Get training failures (for original constraint)
        r_student, g_student = self.get_student_training_log()
        g_student = np.asarray(g_student)
        training_failures = np.count_nonzero(
            g_student[self.student_training_episodes_current_env:, 0] > 0)
        self.student_failures += training_failures
        self.student_training_episodes_current_env = len(r_student)
        return obs, r, done, info

    def compute_obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=float)
        self.test_env = self.final_env
        (rewards, lagrangian_rewards, constraint_values,
         termination, lengths) = self.evaluate_student()
        obs[0] = np.mean(termination == 1)
        obs[1] = self.old_action if self.old_action is not None else 0
        return obs

    def compute_reward(self):
        self.test_env = self.final_env
        (rewards, lagrangian_rewards, constraint_values,
         termination, lengths) = self.evaluate_student()

        # Use custom reward that uses normal reward if there is no failure and n * reward for timeout for failure
        custom_rewards = rewards.copy()
        custom_rewards[termination == -1] = 2 * -0.1 * self.test_episode_timeout

        m = custom_rewards.mean()

        if self.student_success_metric is None:
            self.student_success_metric = m
            return m
        else:
            r = m - self.student_success_metric
            self.student_success_metric = m
            return r


class DangerousGridWorldTrainingObservation(DangerousGridWorldTeacherEnv):
    def __init__(self,student_cls, student_default_kwargs, interventions,
                 final_env, logger_cls, student_ranges_dict={},
                 learning_steps=4000, test_episode_number=20,
                 test_episode_timeout=200, normalize_obs=True,
                 time_steps_lim=np.inf, rounds_lim=np.inf, cost_lim=np.inf):
        super().__init__(student_cls, student_default_kwargs, interventions,
                     final_env, logger_cls, student_ranges_dict,
                     learning_steps, test_episode_number,
                     test_episode_timeout, normalize_obs,
                     time_steps_lim, rounds_lim, cost_lim)
        self.observation_space = gym.spaces.Box(low=np.inf, high=-np.inf,
                                                shape=(2,), dtype=np.float)

    def compute_obs(self):
        r, g = self.get_student_training_log()

        if r is None or g is None:
            student_training_r = 0
            student_training_interventions = 0
        else:
            start = self.student_training_episodes_current_env
            student_training_r = np.mean(r[start:])
            student_training_interventions = np.mean(
                np.sum(np.asarray(g)[start:, 1:], axis=1))

        return np.array([student_training_r, student_training_interventions])

    def compute_reward(self):
        return 0


def train_a_teacher():
    kern = GPy.kern.RBF(input_dim=7, variance=1,
                        lengthscale=[1., 0.05, 1, 0.05, 0.5, 0.5, 0.5],
                        ARD=True)
    kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(1, 1),
                                np.array([0, 2]))
    kern.lengthscale.priors.add(GPy.priors.Gamma.from_EV(0.05, 0.02),
                                np.array([1, 3]))
    kern.lengthscale.priors.add(GPy.priors.Uniform(0, 4),
                                np.array([4, 5, 6]))
    kern.variance.set_prior(GPy.priors.Gamma.from_EV(1, 0.2))
    model = GPModel(kernel=kern, noise_var=0.05, max_iters=1000)

    teacher_env = create_teacher_env(obs_from_training=True)
    student_final_env = base_cenv_fn()

    def init_teaching_policy(params, name=None):
        params = np.squeeze(np.array(params))
        thresholds = params[:4]
        thresholds = thresholds.reshape(2, 2)
        available_actions = params[4:].astype(np.int64)
        policy = SingleSwitchPolicy(thresholds, available_actions, name=name)
        return policy

    def bo_objective(params):
        policy = init_teaching_policy(params)
        return evaluate_single_switch_policy(policy, teacher_env,
                                             student_final_env)

    # Logging dir
    exp_starting_time = datetime.now().strftime('%d_%m_%y__%H_%M_%S')
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               os.pardir, os.pardir, os.pardir, 'results',
                               'dangerous_grid_world')
    base_dir = os.path.join(results_dir, 'teacher_training', exp_starting_time)
    os.makedirs(base_dir, exist_ok=True)

    my_bo = BayesianOptimization(bo_objective,
                                 domain=domain,
                                 initial_design_numdata=10,
                                 initial_design_type='random',
                                 acquisition_type='LCB',
                                 maximize=True,
                                 normalize_Y=True,
                                 model_update_interval=1,
                                 verbosity=True,
                                 model=model)

    my_bo.suggest_next_locations()  # Creates the GP model
    my_bo.model.model['Gaussian_noise.variance'].set_prior(
        GPy.priors.Gamma.from_EV(0.01, 0.1))

    t = time.time()
    my_bo.run_optimization(20,
                           verbosity=True,
                           report_file=os.path.join(base_dir, 'bo_report.txt'),
                           evaluations_file=os.path.join(base_dir,
                                                         'bo_evaluations.csv'),
                           models_file=os.path.join(base_dir, 'bo_model.csv'))
    print(f'Optimization complete in {time.time() - t}')
    print(f'Optimal threshold: {my_bo.x_opt}')
    print(f'Optimal return: {my_bo.fx_opt}')
    np.savez(os.path.join(base_dir, 'solution.npz'), xopt=my_bo.x_opt,
             fxopt=my_bo.fx_opt)
    trained_policy = init_teaching_policy(my_bo.x_opt)
    save_path = os.path.join(base_dir, 'trained_teacher')
    trained_policy.save(save_path)


if __name__ == "__main__":
    train_a_teacher()
