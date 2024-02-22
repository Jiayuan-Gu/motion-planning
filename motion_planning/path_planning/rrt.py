# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import logging
from typing import Callable, List

import numpy as np

from .core import StateSpace

logger = logging.getLogger("pymp.rrt")


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent


class RRT:
    state_space: StateSpace

    def __init__(self, state_space: StateSpace, goal_bias):
        self.state_space = state_space
        self.goal_bias = goal_bias

    def setup(
        self,
        start_states,
        goal_sampler: Callable,
        is_goal_satisfied_fn: Callable,
        max_dist,
        max_iter,
        seed=None,
    ):
        self.start_states = start_states
        self.goal_sampler = goal_sampler
        self.is_goal_satisfied_fn = is_goal_satisfied_fn
        self.max_dist = max_dist
        self.max_iter = max_iter

        self._rng = np.random.RandomState(seed)
        self._nodes: List[Node] = []
        self._n_iter = 0

        self.status = None

    def solve(self):
        solution = None

        # Add start states into nodes
        for start_state in self.start_states:
            if self.check_state_validity(start_state):
                node = Node(start_state)
                self._nodes.append(node)

        if len(self._nodes) == 0:
            logger.debug("There are no valid initial states!")
            self.status = "invalid start"
            return None

        while not self.should_terminate():
            # Sample random state (with goal biasing)
            if self._rng.rand() < self.goal_bias:
                rstate = self.goal_sampler()
            else:
                rstate = self.sample_uniform()

            # Find closest state in the tree
            nnode = self.get_nearest_node(rstate)
            nstate = nnode.state
            dstate = rstate

            # Find state to add
            dist = self.state_space.compute_distance(nstate, rstate)
            if dist > self.max_dist:
                dstate = self.state_space.interpolate(
                    nstate, rstate, self.max_dist / dist
                )

            # TODO(jigu): check motion from nstate to dstate
            if self.check_state_validity(dstate):
                logger.info(self._n_iter)
                node = Node(dstate, parent=nnode)
                self._nodes.append(node)
            else:
                continue

            # TODO(jigu): approximate solution
            if self.is_goal_satisfied_fn(dstate):
                logger.debug("Find the goal after {} iterations".format(self._n_iter))
                solution = node
                break

        path = []
        while solution is not None:
            path.append(solution.state)
            solution = solution.parent

        if len(path) == 0:
            self.status = "failure"
            return None
        else:
            self.status = "success"
            return path[::-1]

    def check_state_validity(self, state) -> bool:
        self._n_iter += 1
        return self.state_space.check_validity(state)

    def should_terminate(self):
        return self._n_iter >= self.max_iter

    def sample_uniform(self):
        return self.state_space.sample_uniform(self._rng)

    def get_nearest_node(self, state) -> Node:
        node_states = [node.state for node in self._nodes]
        dist = self.state_space.compute_distances(state, node_states)
        return self._nodes[np.argmin(dist)]

    def is_goal_satisfied(self, state) -> bool:
        return self.state_space.compute_distance(state, self.goal) <= self.goal_thresh
