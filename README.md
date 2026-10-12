"""
DeepSearch: Simplified Tree-Based Search with Verifiable Rewards
----------------------------------------------------------------
Inspired by the research paper:
"DeepSearch: Overcome the Bottleneck of Reinforcement Learning
with Verifiable Rewards via Tree-Based Search"

Educational prototype only. This does not train an LLM or fully
reproduce the original research paper.

Run:
    python deepsearch_midterm.py
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random


# ======================== CONFIGURATION ========================

START_STATE = 1
TARGET_STATE = 10
MAX_DEPTH = 5
SEARCH_ITERATIONS = 80
RANDOM_SEED = 7

ACTIONS = ["+1", "+2", "*2"]


# ======================== TASK AND REWARD ======================

def apply_action(state: int, action: str) -> int:
    """Apply an action to the current state."""
    if action == "+1":
        return state + 1
    elif action == "+2":
        return state + 2
    elif action == "*2":
        return state * 2
    else:
        raise ValueError(f"Unknown action: {action}")


def verify_answer(state: int) -> bool:
    """Check whether the target state has been reached."""
    return state == TARGET_STATE


def reward_for_state(state: int) -> float:
    """Give reward 1 for success and 0 otherwise."""
    return 1.0 if verify_answer(state) else 0.0


def heuristic(state: int) -> float:
    """Estimate how close a state is to the target."""
    return 1.0 / (1.0 + abs(TARGET_STATE - state))


# =========================== TREE NODE =========================

@dataclass
class Node:
    state: int
    path: List[str] = field(default_factory=list)
    states: List[int] = field(default_factory=list)
    parent: Optional["Node"] = None
    visits: int = 0
    total_reward: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)
    untried_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.untried_actions:
            self.untried_actions = ACTIONS.copy()


def mean_reward(node: Node) -> float:
    """Calculate the average reward of a node."""
    if node.visits == 0:
        return 0.0
    return node.total_reward / node.visits


# =========================== TREE SEARCH =======================

def collect_frontier(root: Node) -> List[Node]:
    """Find nodes that can still be expanded."""
    frontier = []
    stack = [root]

    while stack:
        node = stack.pop()

        finished = (
            verify_answer(node.state)
            or len(node.path) >= MAX_DEPTH
        )

        if not finished and node.untried_actions:
            frontier.append(node)

        stack.extend(node.children.values())

    return frontier


def choose_global_frontier(frontier: List[Node]) -> Node:
    """
    Select a frontier node using a heuristic and exploration bonus.
    This is a simplified selection rule, not standard UCT.
    """

    def score(node: Node) -> float:
        exploration_bonus = 1.0 / ((node.visits + 1) ** 0.5)
        return heuristic(node.state) + exploration_bonus

    return max(frontier, key=score)


def expand(node: Node, rng: random.Random) -> Optional[Node]:
    """Expand a node using one of its untried actions."""
    if not node.untried_actions:
        return None

    action = rng.choice(node.untried_actions)
    node.untried_actions.remove(action)

    next_state = apply_action(node.state, action)

    child = Node(
        state=next_state,
        path=node.path + [action],
        states=node.states + [next_state],
        parent=node,
    )

    node.children[action] = child
    return child


def backpropagate(node: Node, reward: float) -> None:
    """
    Update visit counts and accumulated rewards from the selected
    node back to the root. This does not update model parameters.
    """
    current = node

    while current is not None:
        current.visits += 1
        current.total_reward += reward
        current = current.parent


# ========================== REPLAY BUFFER ======================

def path_signature(path: List[str]) -> Tuple[str, ...]:
    """Convert an action sequence into a hashable signature."""
    return tuple(path)


def add_to_replay_buffer(
    replay_buffer: List[dict],
    seen_paths: set,
    node: Node,
) -> bool:
    """Store a successful trajectory only once."""

    if not verify_answer(node.state):
        return False

    signature = path_signature(node.path)

    if signature in seen_paths:
        return False

    seen_paths.add(signature)

    replay_buffer.append({
        "actions": node.path.copy(),
        "states": node.states.copy(),
        "final_state": node.state,
        "reward": reward_for_state(node.state),
    })

    return True


# ============================= MAIN ============================

def run_search() -> List[dict]:
    rng = random.Random(RANDOM_SEED)

    root = Node(
        state=START_STATE,
        path=[],
        states=[START_STATE],
    )

    replay_buffer = []
    seen_paths = set()

    print("=" * 65)
    print("DeepSearch: Simplified Search Demonstration")
    print("=" * 65)

    print(f"Start state       : {START_STATE}")
    print(f"Target state      : {TARGET_STATE}")
    print(f"Available actions : {ACTIONS}")
    print(f"Maximum depth     : {MAX_DEPTH}")
    print(f"Search iterations : {SEARCH_ITERATIONS}")
    print(f"Random seed       : {RANDOM_SEED}")
    print()

    # -------------------- Search iterations --------------------

    for iteration in range(1, SEARCH_ITERATIONS + 1):

        frontier = collect_frontier(root)

        if not frontier:
            print(
                f"Search stopped at iteration {iteration}: "
                "no expandable nodes remain."
            )
            break

        # Select a promising node.
        selected = choose_global_frontier(frontier)

        # Expand the selected node.
        child = expand(selected, rng)

        if child is None:
            continue

        # Verify the result and calculate its reward.
        reward = reward_for_state(child.state)

        # Save a successful trajectory.
        if verify_answer(child.state):
            add_to_replay_buffer(
                replay_buffer,
                seen_paths,
                child,
            )

        # Propagate reward through the tree.
        backpropagate(child, reward)

        # Display early iterations and successful discoveries.
        if iteration <= 10 or reward > 0:
            print(
                f"Iteration {iteration:02d} | "
                f"{selected.state} --{child.path[-1]}--> "
                f"{child.state} | "
                f"Reward = {reward:.0f} | "
                f"Frontier = {len(frontier)}"
            )

    # ---------------------- Final results ----------------------

    print()
    print("=" * 65)
    print("SEARCH SUMMARY")
    print("=" * 65)

    print(f"Root visits                   : {root.visits}")
    print(
        f"Unique successful trajectories: "
        f"{len(replay_buffer)}"
    )
    print(f"Replay buffer size             : {len(replay_buffer)}")

    if replay_buffer:
        print("\nVerified successful trajectories:")

        for index, item in enumerate(replay_buffer, start=1):
            state_path = " -> ".join(map(str, item["states"]))
            action_path = " -> ".join(item["actions"])

            print(f"\nTrajectory {index}")
            print(f"States  : {state_path}")
            print(f"Actions : {action_path}")
            print(f"Reward  : {item['reward']:.0f}")

    else:
        print("\nNo successful trajectory was found.")

    # ------------------------ Limitations ----------------------

    print("\nLIMITATIONS")
    print("-" * 65)
    print("1. The problem uses numerical states, not language models.")
    print("2. The node-selection method is simplified, not standard UCT.")
    print("3. The replay buffer stores paths but does not train a model.")
    print("4. This is not a full reproduction of the DeepSearch paper.")
    print("5. The results do not establish improvements in AI reasoning.")

    print("=" * 65)

    return replay_buffer


if __name__ == "__main__":
    run_search()
