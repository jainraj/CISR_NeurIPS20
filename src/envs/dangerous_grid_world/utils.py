from src.envs.dangerous_grid_world.dangerous_grid_world_custom import DangerousGridWorldEnvCustom
import numpy as np
from src.envs.CMDP import CMDP
from src.envs.dangerous_grid_world.dangerous_grid_world_constants import TERMINAL
from collections import Counter
from matplotlib import pyplot as plt


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

    if n_steps < 1:
        reach_mask = np.copy(base_danger_mask)
    else:
        previous_mask = np.copy(base_danger_mask)
        for _ in range(n_steps):
            reach_mask = one_step_reach(previous_mask)
            previous_mask[:] |= reach_mask[:]

    intervention_states = []
    xs, ys = reach_mask.nonzero()
    for x, y in zip(xs, ys):
        intervention_states.append(np.ravel_multi_index((x, y), DangerousGridWorldEnvCustom().grid_specs))

    return intervention_states


if __name__ == "__main__":
    get_intervention_states(0)



def create_intervention_from_list(intervention_states):

    def intervention_condition(env, **kwargs):
        return env.get_state() in intervention_states

    return intervention_condition


def deploy(model, env, timesteps=1000):
    obs = env.reset()
    reward_sum, length, successes, n_episodes = (0.0, 0, 0, 0)
    returns, returns_success, trajectories, trajectory = ([], [], [], [])

    for _ in range(timesteps):
        action, _ = model.predict(obs, deterministic=False)
        if isinstance(env, CMDP):
            obs, reward, g, done, info = env.step(action)
        else:
            obs, reward, done, info = env.step(action)
        reward_sum += reward
        length += 1
        trajectory.append(env.s)
        if done:
            success = info['next_state_type'] == TERMINAL
            successes += float(success)
            returns.append(reward_sum)
            if success:
                returns_success.append(reward_sum)
            length = 0
            reward_sum = 0.0
            n_episodes += 1
            obs = env.reset()
            trajectories.append(trajectory)
            trajectory = []
    if trajectory:
        trajectories.append(trajectory)
    if n_episodes == 0:
        n_episodes = 1
        returns.append(reward_sum)
    success_ratio = successes / n_episodes
    avg_return = np.mean(returns)
    avg_return_success = np.mean(returns_success)
    return success_ratio, avg_return, avg_return_success, trajectories


def plot_trajectories(traj, world_shape):
    plt.figure(figsize=(20, 10))
    intensity_map = np.zeros(world_shape, dtype=int)
    occupancy = Counter(np.hstack(traj))
    for k, v in occupancy.items():
        intensity_map[np.unravel_index(k, world_shape)] = v
    plt.imshow(intensity_map)
    for t in traj:
        plt.plot(*np.unravel_index(t, world_shape)[::-1])
