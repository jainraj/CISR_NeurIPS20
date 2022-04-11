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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def constraint(info=None, **kwargs):
    return {'g': float(info['next_state_type'] == 'danger')}


def base_cenv_fn():
    # get the base constrained env
    return CMDP(DangerousGridWorldEnvCustom(timeout=200), constraint,
                constraints_values=[0],
                n_constraints=1,
                avg_constraint=True)


def make_base_cenvs():
    # make intervention base constrained envs
    dist = [1, 1]
    tau = [0.1, 0]
    buff_size = [1, 0]
    avg_constraint = [True, True]
    interventions = []

    for d, t, b, avg in zip(dist, tau, buff_size, avg_constraint):
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
    learning_steps = 4800 * 2
    time_steps_lim = learning_steps * 10
    test_episode_timeout = 200
    test_episode_number = 5

    # todo: keep same for now
    if obs_from_training:
        env_cls = TeacherEnv
    elif non_stationary_bandit:
        env_cls = TeacherEnv
    else:
        env_cls = TeacherEnv

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
