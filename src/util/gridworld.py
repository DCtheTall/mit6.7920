"""
Implementation of GridWorld MDP
===============================
Implementation of a GirdWorld Markov Decision Process (MDP)
for demonstrating different learning algorithms.

"""

import enum
import numpy as np
from typing import Callable, Dict, List, Set, Tuple


N_ACTIONS = 4


class Action(enum.Enum):
    __order__ = 'Up Left Down Right'
    Up = 0
    Left = 1
    Down = 2
    Right = 3


State = Tuple[int, int]
States = Set[State]
Actions = List[Action]
Transitions = Dict[Tuple[State, Action, State], float]
Rewards = Dict[State, float]


class GridWorld:
    # Set of all states and useful states
    S: States
    start: State
    goal: State
    failure: State
    # List of all actions
    A: List[Action]
    # Transition probabilities
    P: Transitions
    # Rewards
    R: Rewards

    def __init__(self,
                 size: int,
                 # First is probability you go in direction of action.
                 # Second is probability you go in any other direction
                 # *but* opposite of action.
                 # For static GridWorld use (1.0, 0.0).
                 transition_probs: Tuple[float, float] = (1.0/3.0,) * 2,
                 goal_reward: float = 1.0,
                 failure_reward: float = -1.0):
        self.S = {(i // size, i % size)
                  for i in range(size ** 2)}
        self.start = (0, 0)
        # Upper right corner
        self.goal = (size - 1, size - 1)
        # One below goal
        self.failure = (size - 1, size - 2)
        self.A = [a for a in Action]
        self.P = _build_transition_probabilities(self.S, self.A,
                                                 self.is_terminal_state,
                                                 transition_probs)
        self.R = {s: 0.0 for s in self.S}
        self.R[self.goal] = goal_reward
        self.R[self.failure] = failure_reward

    def step(self, s: State, a: Action) -> State:
        """Sample next state given state and action"""
        possible_states = []
        p_weights = []
        for s_prime in self.S:
            p = self.P.get((s, a, s_prime), 0.0)
            if p > 0.0:
                possible_states.append(s_prime)
                p_weights.append(p)
        assert np.isclose(sum(p_weights), 1.0)
        idx = np.random.choice([i for i, _ in enumerate(possible_states)],
                               size=1,
                               p=p_weights)[0]
        return possible_states[idx]
    
    def is_terminal_state(self, s: State) -> bool:
        return s in {self.goal, self.failure}


def _build_transition_probabilities(
        S: States,
        A: Actions,
        is_terminal: Callable[[State], bool],
        transition_probs: Tuple[float, float]) -> Transitions:
    p1, p2 = transition_probs
    assert np.isclose(1.0, p1 + 2 * p2)
    P = {}
    for s in S:
        for a in A:
            if is_terminal(s):
                P[(s, a, s)] = 1.0
                continue
            possible_next_states = []
            p_weights = []
            for s_prime in S:
                dx, dy = s_prime[0] - s[0], s_prime[1] - s[1]
                if max(abs(dx), abs(dy), abs(dx) + abs(dy)) != 1:
                    continue
                if a == Action.Left:
                    if dx == 1:
                        continue
                    p_weights.append(p1 if dx == -1 else p2)
                if a == Action.Right:
                    if dx == -1:
                        continue
                    p_weights.append(p1 if dx == 1 else p2)
                if a == Action.Up:
                    if dy == -1:
                        continue
                    p_weights.append(p1 if dy == 1 else p2)
                if a == Action.Down:
                    if dy == 1:
                        continue
                    p_weights.append(p1 if dy == -1 else p2)
                possible_next_states.append(s_prime)
            for s_prime, p_weight in zip(possible_next_states, p_weights):
                P[(s, a, s_prime)] = p_weight
            if len(p_weights) < 3:
                P[(s, a, s)] = 1.0 - sum(p_weights)
    return P
