# Copyright (c) 2022 Jiayuan Gu
# Licensed under The MIT License [see LICENSE for details]

import logging
from typing import Iterable, List

import numpy as np

from .core import StateSpace

logger = logging.getLogger("pymp.rrt_connect")


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

    def traceback(self):
        node = self
        path = []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return path


class RRTConnect:
    state_space: StateSpace

    def __init__(self, state_space: StateSpace):
        self.state_space = state_space

    def setup(
        self,
        start_states,
        goal_iter: Iterable,
        max_dist,
        max_iter,
        start_state_range,
        start_state_max_trials,
        seed=None,
    ):
        self.start_states = start_states
        self.goal_iter = goal_iter
        self.max_dist = max_dist
        self.max_iter = max_iter
        self.start_state_range = start_state_range
        self.start_state_max_trials = start_state_max_trials

        self._rng = np.random.RandomState(seed)
        self._start_tree: List[Node] = []
        self._goal_tree: List[Node] = []
        self._n_iter = 0

        self.status = None

    def solve(self):
        # Add start states into tree
        for start_state in self.start_states:
            if self.check_state_validity(start_state):
                node = Node(start_state)
                self._start_tree.append(node)

        if len(self._start_tree) == 0:
            if self.start_state_range == 0.0:
                logger.info("There are no valid initial states!")
                self.status = "invalid start"
                return None

            logger.info(
                "There are no valid initial states! Try to sample nearby initial states."
            )

            # NOTE(jigu): Sampling in the loop might be inefficient,
            # but can support vector format of start_state_range
            for _ in range(self.start_state_max_trials):
                offset = self._rng.uniform(
                    -self.start_state_range, self.start_state_range
                )
                nearby_start_state = start_state + offset
                if self.check_state_validity(nearby_start_state):
                    self._start_tree.append(Node(nearby_start_state))
                    # break

            if len(self._start_tree) == 0:
                logger.info("There are no valid (nearby) initial states!")
                self.status = "invalid start"
                return None

        # Add goal states into tree
        for goal_state in self.goal_iter:
            if self.check_state_validity(goal_state):
                node = Node(goal_state)
                self._goal_tree.append(node)

        if len(self._goal_tree) == 0:
            logger.info("There are no valid goal states!")
            self.status = "invalid goal"
            return None

        # A flag that toggles between expanding the start tree (true) or goal tree (false)
        is_start_tree = False

        while not self.should_terminate():
            is_start_tree = not is_start_tree
            tree = self._start_tree if is_start_tree else self._goal_tree
            other_tree = self._goal_tree if is_start_tree else self._start_tree

            # Sample random state
            rstate = self.sample_uniform()

            # From current tree to other tree
            node, status = self.grow_tree(tree, rstate)

            # Try another random state to grow tree
            if status == "TRAPPED":
                continue

            # Attempt to connect trees
            other_node, status = self.grow_tree(other_tree, node.state)
            while status == "ADVANCED":
                other_node, status = self.grow_tree(
                    other_tree, node.state, nnode=other_node
                )

            # If we connected the trees in a valid way
            if status == "REACHED":
                logger.debug("Find solution at %d steps", self._n_iter)
                path = node.traceback()[::-1] + other_node.traceback()
                if not is_start_tree:
                    path = path[::-1]
                self.status = "success"
                return path
        else:
            self.status = "failure"
            return []

    def check_state_validity(self, state) -> bool:
        self._n_iter += 1
        return self.state_space.check_validity(state)

    def should_terminate(self):
        return self._n_iter >= self.max_iter

    def sample_uniform(self):
        return self.state_space.sample_uniform(self._rng)

    def get_nearest_node(self, tree: List[Node], state) -> Node:
        node_states = [node.state for node in tree]
        dist = self.state_space.compute_distances(state, node_states)
        return tree[np.argmin(dist)]

    def grow_tree(self, tree, rstate, add_node=True, nnode=None):
        if nnode is None:
            # Find closest state in the tree
            nnode = self.get_nearest_node(tree, rstate)
        nstate = nnode.state

        # Assume we can reach the state we go towards
        reach = True

        # Find state to add
        dstate = rstate
        dist = self.state_space.compute_distance(nstate, rstate)
        if dist > self.max_dist:
            dstate = self.state_space.interpolate(nstate, rstate, self.max_dist / dist)
            reach = False

        is_valid = self.check_state_validity(dstate)
        if not is_valid:
            return None, "TRAPPED"

        node = Node(dstate, parent=nnode)
        if add_node:
            tree.append(node)
        return node, ("REACHED" if reach else "ADVANCED")
