from src.envs.dangerous_grid_world.dangerous_grid_world_custom import DangerousGridWorldEnvCustom
import numpy as np


def one_step_reach(base_mask):
    """
    One step reachability operator for grid world with four actions.

    Given a mask of starting states
    [[False, False, False],
     [False, True,  False],
     [False, False, False]]

    returns the mask of one step reachable states
    [[False, True,  False],
     [True,  False, True ],
     [False, True,  False]]
    """
    n, m = base_mask.shape

    one_step_mask = np.zeros_like(base_mask, dtype=bool)

    # Move right and left
    one_step_mask |= np.hstack((np.zeros((n, 1), dtype=bool), base_mask[:, :-1]))
    one_step_mask |= np.hstack((base_mask[:, 1:], np.zeros((n, 1), dtype=bool)))
    # Move up and down
    one_step_mask |= np.vstack((np.zeros((1, m), dtype=bool), base_mask[:-1, :]))
    one_step_mask |= np.vstack((base_mask[1:, :], np.zeros((1, m), dtype=bool)))

    return one_step_mask


def get_intervention_states(n_steps=1):
    base_danger_mask = DangerousGridWorldEnvCustom().danger_states

    previous_mask = np.copy(base_danger_mask)
    for _ in range(n_steps):
        reach_mask = one_step_reach(previous_mask)
        previous_mask[:] |= reach_mask[:]

    intervention_states = []
    for x, y in reach_mask.nonzero():
        intervention_states.append(np.ravel_multi_index((x, y), DangerousGridWorldEnvCustom().grid_specs))

    return intervention_states


def create_intervention_from_list(intervention_states):

    def intervention_condition(env, **kwargs):
        return env.get_state() in intervention_states

    return intervention_condition
