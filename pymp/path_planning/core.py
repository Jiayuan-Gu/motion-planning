from typing import Callable

import numpy as np


class StateSpace:
    def sample_uniform(self, rng: np.random.RandomState):
        raise NotImplementedError

    def compute_distance(self, state0, state1):
        raise NotImplementedError

    def compute_distances(self, state0, state1):
        # Assume compute_distance supports broadcast
        return self.compute_distance(state0, state1)

    def interpolate(self, state0, state1, weight):
        raise NotImplementedError

    def set_state_validity_checker(self, fn: Callable[..., bool]):
        self.state_validity_checker = fn

    def check_validity(self, state):
        return self.state_validity_checker(state)


class GoalState:
    def __init__(self, goal, state_space: StateSpace, threshold, seed=None):
        self.goal = goal
        self.state_space = state_space
        self.threshold = threshold
        self.rng = np.random.RandomState(seed)

    def sample(self):
        return self.goal

    def is_satisfied(self, state):
        return self.state_space.compute_distance(state, self.goal) <= self.threshold


class JointStateSpace(StateSpace):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample_uniform(self, rng=np.random):
        return rng.uniform(self.low, self.high)

    def compute_distance(self, state0, state1):
        return np.linalg.norm(state1 - state0, axis=-1)

    def interpolate(self, state0, state1, weight):
        return state0 + (state1 - state0) * weight

    def get_goal_sampler(self):
        pass

    def get_is_goal_satisfied_fn(self, goal, thresh):
        goal = np.array(goal)

        def is_goal_satisfied_fn(state):
            return np.any(np.linalg.norm(goal - state, axis=-1) <= thresh)

        return is_goal_satisfied_fn


class GoalStates(GoalState):
    def sample(self):
        ind = self.rng.choice(len(self.goal))
        return self.goal[ind]

    def is_satisfied(self, state):
        return np.any(
            self.state_space.compute_distances(state, self.goal) <= self.threshold
        )
