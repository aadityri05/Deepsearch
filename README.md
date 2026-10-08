# Deepsearch
"""
DeepSearch-style RLVR prototype implemented from scratch with Python's standard library.

This is an independent algorithmic implementation inspired by the DeepSearch paper's
components:
  1. MCTS-style reasoning tree construction
  2. Entropy/confidence-guided negative trajectory selection
  3. Global frontier selection
  4. Constrained Q-value backup
  5. Progressive hard-example filtering
  6. Replay-buffer solution caching
  7. A small Tree-GRPO-style policy update on a tabular policy

Important:
- This file does NOT copy implementation code from the paper.
- It is a self-contained research prototype, not a reproduction of a 1.5B LLM.
- The policy is tabular so the complete algorithm can run locally without GPU/model weights.
- A real LLM can replace TabularPolicy by implementing the same interface.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class DeepSearchConfig:
    # Search
    max_depth: int = 8
    expansion_width: int = 8
    uct_lambda: float = 2.0
    lambda1: float = 0.4
    lambda2: float = 0.0          # Paper's training setup uses entropy coefficient 0 for the actor.
    lambda3: float = 0.01
    min_gamma: float = 0.1
    qmax: float = 1.0
    q_temperature: float = 1.0

    # Adaptive training / replay
    pass_k: int = 4
    hard_threshold: float = 0.25
    direct_rollout_fraction: float = 0.25
    replay_capacity: int = 1000

    # Demo policy update. Deliberately larger than a real LLM LR so the toy example learns.
    learning_rate: float = 0.05
    clip_low: float = 0.20
    clip_high: float = 0.28
    gradient_clip: float = 1.0

    # Search budget for the toy demonstration
    mcts_rollouts: int = 24
    random_seed: int = 7


# Values from the paper's experimental section are kept here for reference.
# They should not be confused with the smaller demo configuration above.
PAPER_CONFIGURATION = {
    "training_gpus": "16 x H100 96GB",
    "base_model": "Nemotron-Research-Reasoning-Qwen-1.5B v2",
    "dataset": "DeepMath-103K",
    "prompt_length": 2048,
    "response_limit": 16384,
    "global_batch_size": 256,
    "uct_lambda": 2.0,
    "max_depth": 64,
    "node_expansion_length": 256,
    "expansion_width": 8,
    "global_lambda1": 0.4,
    "global_lambda3": 0.01,
    "q_temperature": 1.0,
    "qmax": 1.0,
    "incomplete_reward": -1.0,
    "learning_rate": 1e-6,
    "warmup_steps": 10,
    "weight_decay": 0.1,
    "gradient_clip": 1.0,
    "clip_low": 0.20,
    "clip_high": 0.28,
    "clipping_ratio": 10.0,
    "mini_batch_size": 32,
    "max_tokens_per_gpu": 18432,
    "entropy_coefficient": 0.0,
    "kl_regularization": False,
    "sampling_temperature": 1.0,
    "sampling_top_p": 1.0,
    "rollouts_per_prompt": 8,
    "checkpoint_interval": 5,
    "training_steps": 100,
    "filter_threshold": 0.25,
}


# -----------------------------------------------------------------------------
# Toy verifiable reasoning environment
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ArithmeticProblem:
    """Toy problem: reach a target integer from zero using bounded integer moves."""

    problem_id: str
    target: int
    max_steps: int = 6
    allowed_moves: Tuple[int, ...] = (-2, -1, 1, 2, 3)


class ToyVerifier:
    """Binary verifier analogous to a verifiable final-answer reward."""

    @staticmethod
    def verify(problem: ArithmeticProblem, state: int, depth: int) -> int:
        if state == problem.target:
            return 1
        if depth >= problem.max_steps:
            return -1
        return 0


# -----------------------------------------------------------------------------
# Policy: tabular categorical model
# -----------------------------------------------------------------------------

class TabularPolicy:
    """
    Minimal categorical policy.

    State key = (problem_id, current_value, depth).
    Each action is one integer move. The implementation is deliberately small so
    the search/learning logic is transparent.
    """

    def __init__(self, action_space: Sequence[int], seed: int = 7) -> None:
        self.actions = tuple(action_space)
        self.logits: Dict[Tuple[str, int, int], Dict[int, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )
        self.rng = random.Random(seed)

    @staticmethod
    def _softmax(logits: Dict[int, float], temperature: float = 1.0) -> Dict[int, float]:
        temperature = max(temperature, 1e-8)
        scaled = {a: v / temperature for a, v in logits.items()}
        m = max(scaled.values())
        exps = {a: math.exp(v - m) for a, v in scaled.items()}
        z = sum(exps.values())
        return {a: e / z for a, e in exps.items()}

    def probabilities(
        self, problem: ArithmeticProblem, state: int, depth: int, temperature: float = 1.0
    ) -> Dict[int, float]:
        return self._softmax(self.logits[(problem.problem_id, state, depth)], temperature)

    def entropy(self, problem: ArithmeticProblem, state: int, depth: int) -> float:
        p = self.probabilities(problem, state, depth)
        return -sum(prob * math.log(max(prob, 1e-12)) for prob in p.values())

    def sample_actions(
        self,
        problem: ArithmeticProblem,
        state: int,
        depth: int,
        count: int,
        temperature: float = 1.0,
    ) -> List[Tuple[int, float]]:
        p = self.probabilities(problem, state, depth, temperature)
        actions = list(p.keys())
        weights = [p[a] for a in actions]
        samples: List[Tuple[int, float]] = []
        for _ in range(count):
            a = self.rng.choices(actions, weights=weights, k=1)[0]
            samples.append((a, -math.log(max(p[a], 1e-12))))
        return samples

    def log_prob(self, problem: ArithmeticProblem, state: int, depth: int, action: int) -> float:
        p = self.probabilities(problem, state, depth)
        return math.log(max(p[action], 1e-12))

    def reinforce_update(
        self,
        examples: Sequence["TokenTrainingExample"],
        learning_rate: float,
        clip_low: float,
        clip_high: float,
        gradient_clip: float,
    ) -> float:
        """
        Small policy-gradient update with a PPO/DAPO-style ratio clip.

        For each example, advantage is computed from tree q-values. The gradient
        of log pi(a|s) for a categorical softmax is implemented explicitly:
            d log pi(a) / d logit(b) = I[a=b] - pi(b)
        """

        total_loss = 0.0
        grads: Dict[Tuple[str, int, int], Dict[int, float]] = defaultdict(
            lambda: {a: 0.0 for a in self.actions}
        )

        for ex in examples:
            if ex.old_prob <= 0.0:
                continue
            current_prob = math.exp(self.log_prob(ex.problem, ex.state, ex.depth, ex.action))
            ratio = current_prob / max(ex.old_prob, 1e-12)

            # PPO/DAPO-style clipping. The implementation treats the clipped
            # region as zero policy-gradient contribution, matching the usual
            # piecewise objective at the level needed for this prototype.
            clipped = min(max(ratio, 1.0 - clip_low), 1.0 + clip_high)
            use_ratio = ratio
            if ex.advantage >= 0 and ratio > 1.0 + clip_high:
                use_ratio = clipped
            elif ex.advantage < 0 and ratio < 1.0 - clip_low:
                use_ratio = clipped

            p = self.probabilities(ex.problem, ex.state, ex.depth)
            scale = ex.advantage * use_ratio
            key = (ex.problem.problem_id, ex.state, ex.depth)
            for action in self.actions:
                grad_log_prob = (1.0 if action == ex.action else 0.0) - p[action]
                # Negative sign because we minimize -objective.
                grads[key][action] -= scale * grad_log_prob

            total_loss -= math.log(max(current_prob, 1e-12)) * ex.advantage * use_ratio

        for key, grad_by_action in grads.items():
            norm = math.sqrt(sum(g * g for g in grad_by_action.values()))
            scale = min(1.0, gradient_clip / max(norm, 1e-12))
            for action in self.actions:
                self.logits[key][action] -= learning_rate * grad_by_action[action] * scale

        return total_loss / max(len(examples), 1)


# -----------------------------------------------------------------------------
# Search tree
# -----------------------------------------------------------------------------

@dataclass
class TreeNode:
    node_id: int
    problem: ArithmeticProblem
    state: int
    depth: int
    parent: Optional["TreeNode"] = None
    action: Optional[int] = None
    children: List["TreeNode"] = field(default_factory=list)
    visits: int = 0
    q_value: float = 0.0
    terminal_reward: Optional[int] = None
    terminal: bool = False
    entropy: float = 0.0
    token_nll: float = 0.0

    @property
    def is_frontier(self) -> bool:
        return (not self.terminal) and len(self.children) == 0

    def trajectory(self) -> List["TreeNode"]:
        path: List[TreeNode] = []
        node: Optional[TreeNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()
        return path


@dataclass
class TokenTrainingExample:
    problem: ArithmeticProblem
    state: int
    depth: int
    action: int
    advantage: float
    old_prob: float


# -----------------------------------------------------------------------------
# DeepSearch-style MCTS
# -----------------------------------------------------------------------------

class DeepSearchMCTS:
    def __init__(self, policy: TabularPolicy, verifier: ToyVerifier, config: DeepSearchConfig) -> None:
        self.policy = policy
        self.verifier = verifier
        self.config = config
        self.node_counter = 0

    def _new_node(
        self,
        problem: ArithmeticProblem,
        state: int,
        depth: int,
        parent: Optional[TreeNode] = None,
        action: Optional[int] = None,
    ) -> TreeNode:
        self.node_counter += 1
        return TreeNode(
            node_id=self.node_counter,
            problem=problem,
            state=state,
            depth=depth,
            parent=parent,
            action=action,
        )

    def _uct_score(self, child: TreeNode, parent: TreeNode) -> float:
        if child.visits == 0:
            return float("inf")
        mean_q = child.q_value / child.visits
        exploration = self.config.uct_lambda * math.sqrt(
            math.log(max(parent.visits, 1) + 1.0) / child.visits
        )
        return mean_q + exploration

    def _expand_node(self, node: TreeNode) -> List[TreeNode]:
        if node.terminal or node.depth >= node.problem.max_steps:
            return []

        samples = self.policy.sample_actions(
            node.problem,
            node.state,
            node.depth,
            self.config.expansion_width,
            temperature=1.0,
        )
        new_children: List[TreeNode] = []
        for action, nll in samples:
            child_state = node.state + action
            child_depth = node.depth + 1
            reward = self.verifier.verify(node.problem, child_state, child_depth)
            child = self._new_node(
                node.problem,
                child_state,
                child_depth,
                parent=node,
                action=action,
            )
            child.token_nll = nll
            child.entropy = self.policy.entropy(node.problem, node.state, node.depth)
            child.terminal_reward = reward if reward != 0 else None
            child.terminal = reward != 0
            node.children.append(child)
            new_children.append(child)
        return new_children

    def _frontier_score(self, node: TreeNode, terminal_depth: int) -> float:
        parent_q = node.parent.q_value / max(node.parent.visits, 1) if node.parent else 0.0
        quality = math.tanh(parent_q)
        uncertainty = node.entropy
        depth_bonus = node.depth / max(terminal_depth, 1)
        return (
            self.config.lambda1 * quality
            + self.config.lambda2 * uncertainty
            + self.config.lambda3 * math.sqrt(max(depth_bonus, 0.0))
        )

    def _select_global_frontier(self, root: TreeNode) -> Optional[TreeNode]:
        frontier: List[TreeNode] = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node.is_frontier and node.depth < min(root.problem.max_steps, self.config.max_depth):
                frontier.append(node)
            stack.extend(node.children)
        if not frontier:
            return None
        return max(frontier, key=lambda n: self._frontier_score(n, root.problem.max_steps))

    def _find_terminals(self, nodes: Iterable[TreeNode]) -> List[TreeNode]:
        terminals: List[TreeNode] = []
        stack = list(nodes)
        while stack:
            node = stack.pop()
            if node.terminal:
                terminals.append(node)
            stack.extend(node.children)
        return terminals

    def _local_select_child(self, node: TreeNode) -> Optional[TreeNode]:
        if not node.children:
            return None
        return max(node.children, key=lambda child: self._uct_score(child, node))

    def _avg_trajectory_entropy(self, terminal: TreeNode) -> float:
        path = terminal.trajectory()[1:]  # Exclude question/root node.
        if not path:
            return float("inf")
        return sum(node.token_nll for node in path) / len(path)

    def _select_backprop_trajectory(self, terminals: Sequence[TreeNode]) -> Optional[TreeNode]:
        correct = [n for n in terminals if n.terminal_reward == 1]
        if correct:
            # Any verified correct trajectory can contribute. Prefer the shallowest
            # one to limit unnecessary credit assignment in the demo.
            return min(correct, key=lambda n: n.depth)
        incorrect = [n for n in terminals if n.terminal_reward == -1]
        if not incorrect:
            return None
        # Paper-inspired entropy guidance: select the most confident mistake.
        return min(incorrect, key=self._avg_trajectory_entropy)

    def _constrained_backup(self, trajectory_end: TreeNode) -> None:
        reward = trajectory_end.terminal_reward
        if reward not in (-1, 1):
            return

        path = trajectory_end.trajectory()[1:]  # intermediate/terminal reasoning nodes
        terminal_index = len(path)
        trajectory_ids = {n.node_id for n in path}

        for index, node in enumerate(path, start=1):
            gamma = max(index / max(terminal_index, 1), self.config.min_gamma)
            old_q = node.q_value
            old_sign_product = old_q * reward

            if old_sign_product >= 0:
                node.q_value = old_q + gamma * reward
            elif reward > 0:
                node.q_value = gamma * reward
            elif old_q > 0:
                # Preserve a node that has been observed on a correct path.
                node.q_value = old_q
            else:
                node.q_value = old_q + gamma * reward

            node.q_value = max(-self.config.qmax, min(self.config.qmax, node.q_value))
            node.visits += 1

        # Mark the terminal visit as well.
        trajectory_end.visits += 1
        if trajectory_end.node_id not in trajectory_ids:
            trajectory_end.q_value = float(reward)

    def search(self, problem: ArithmeticProblem) -> Tuple[TreeNode, List[TreeNode], List[TreeNode]]:
        self.node_counter = 0
        root = self._new_node(problem, state=0, depth=0)
        all_selected: List[TreeNode] = []

        for rollout in range(self.config.mcts_rollouts):
            frontier = root if rollout == 0 else self._select_global_frontier(root)
            if frontier is None:
                break

            # Global frontier selection decides where to spend the next budget.
            # Local UCT then chooses among siblings while extending that subtree.
            current = frontier
            newly_created: List[TreeNode] = []
            while not current.terminal and current.depth < min(problem.max_steps, self.config.max_depth):
                children = self._expand_node(current)
                if not children:
                    break
                newly_created.extend(children)
                current.visits += 1
                current = self._local_select_child(current)
                if current is None:
                    break

            # Entropy-guided supervision is restricted to terminals produced by
            # this rollout, rather than retrospectively searching the entire tree.
            rollout_terminals = self._find_terminals(newly_created)
            if current is not None and current.terminal:
                rollout_terminals.append(current)
            unique: Dict[int, TreeNode] = {}
            for node in rollout_terminals:
                unique[node.node_id] = node
            rollout_terminals = list(unique.values())

            terminal = self._select_backprop_trajectory(rollout_terminals)
            if terminal is not None:
                all_selected.append(terminal)
                self._constrained_backup(terminal)

        terminals = self._find_terminals([root])
        return root, terminals, all_selected


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._data: Dict[str, List[int]] = {}

    def add(self, problem_id: str, actions: Sequence[int]) -> None:
        if not actions:
            return
        self._data[problem_id] = list(actions)
        if len(self._data) > self.capacity:
            oldest = next(iter(self._data))
            del self._data[oldest]

    def has(self, problem_id: str) -> bool:
        return problem_id in self._data

    def get(self, problem_id: str) -> Optional[List[int]]:
        return list(self._data[problem_id]) if problem_id in self._data else None

    def __len__(self) -> int:
        return len(self._data)


# -----------------------------------------------------------------------------
# Direct rollouts / Pass@K
# -----------------------------------------------------------------------------

class RolloutEngine:
    def __init__(self, policy: TabularPolicy, verifier: ToyVerifier, seed: int = 7) -> None:
        self.policy = policy
        self.verifier = verifier
        self.rng = random.Random(seed)

    def rollout(self, problem: ArithmeticProblem) -> Tuple[List[int], int]:
        state = 0
        actions: List[int] = []
        for depth in range(problem.max_steps):
            sampled = self.policy.sample_actions(problem, state, depth, 1)[0]
            action = sampled[0]
            actions.append(action)
            state += action
            reward = self.verifier.verify(problem, state, depth + 1)
            if reward != 0:
                return actions, reward
        return actions, -1

    def pass1_at_k(self, problem: ArithmeticProblem, k: int) -> float:
        success = 0
        for _ in range(k):
            _, reward = self.rollout(problem)
            if reward == 1:
                success += 1
        return success / max(k, 1)


# -----------------------------------------------------------------------------
# Tree-to-training conversion and trainer
# -----------------------------------------------------------------------------

class DeepSearchTrainer:
    def __init__(self, problems: Sequence[ArithmeticProblem], config: Optional[DeepSearchConfig] = None) -> None:
        self.config = config or DeepSearchConfig()
        random.seed(self.config.random_seed)
        action_space = tuple(sorted({move for p in problems for move in p.allowed_moves}))
        self.policy = TabularPolicy(action_space, seed=self.config.random_seed)
        self.verifier = ToyVerifier()
        self.mcts = DeepSearchMCTS(self.policy, self.verifier, self.config)
        self.rollouts = RolloutEngine(self.policy, self.verifier, seed=self.config.random_seed)
        self.replay = ReplayBuffer(self.config.replay_capacity)
        self.problems = list(problems)

    def initial_hard_set(self) -> List[ArithmeticProblem]:
        return [
            p
            for p in self.problems
            if self.rollouts.pass1_at_k(p, self.config.pass_k) < self.config.hard_threshold
        ]

    def _trajectory_to_examples(
        self,
        terminal: TreeNode,
        tree_terminals: Sequence[TreeNode],
    ) -> List[TokenTrainingExample]:
        path = terminal.trajectory()[1:]
        terminal_rewards = [n.terminal_reward for n in tree_terminals if n.terminal_reward in (-1, 1)]
        baseline = sum(terminal_rewards) / max(len(terminal_rewards), 1)
        examples: List[TokenTrainingExample] = []
        for node in path:
            q = math.tanh(node.q_value / max(self.config.q_temperature, 1e-8)) * self.config.qmax
            advantage = q - baseline
            old_prob = math.exp(
                self.policy.log_prob(node.problem, node.parent.state if node.parent else 0,
                                     node.parent.depth if node.parent else 0, node.action)
            )
            examples.append(
                TokenTrainingExample(
                    problem=node.problem,
                    state=node.parent.state if node.parent else 0,
                    depth=node.parent.depth if node.parent else 0,
                    action=node.action,
                    advantage=advantage,
                    old_prob=old_prob,
                )
            )
        return examples

    def train_round(self, hard_set: Sequence[ArithmeticProblem]) -> Dict[str, float]:
        training_examples: List[TokenTrainingExample] = []
        correct_found = 0
        search_count = 0

        for problem in hard_set:
            cached = self.replay.get(problem.problem_id)
            if cached is not None:
                # Cached verified solution: directly supervise it, plus a small number
                # of fresh direct rollouts, analogous to the paper's hybrid rollout path.
                state = 0
                for depth, action in enumerate(cached):
                    p = self.policy.probabilities(problem, state, depth)[action]
                    training_examples.append(
                        TokenTrainingExample(problem, state, depth, action, 1.0, p)
                    )
                    state += action
                for _ in range(max(1, int(self.config.mcts_rollouts * self.config.direct_rollout_fraction))):
                    actions, reward = self.rollouts.rollout(problem)
                    state = 0
                    for depth, action in enumerate(actions):
                        p = self.policy.probabilities(problem, state, depth)[action]
                        adv = 1.0 if reward == 1 else -0.5
                        training_examples.append(TokenTrainingExample(problem, state, depth, action, adv, p))
                        state += action
                continue

            search_count += 1
            root, terminals, selected = self.mcts.search(problem)
            correct = [n for n in terminals if n.terminal_reward == 1]
            if correct:
                correct_found += 1
                best = min(correct, key=lambda n: n.depth)
                self.replay.add(problem.problem_id, [n.action for n in best.trajectory()[1:] if n.action is not None])

            # Positive trajectory when available; otherwise a confident negative trajectory.
            chosen = min(correct, key=lambda n: n.depth) if correct else (
                min(
                    [n for n in terminals if n.terminal_reward == -1],
                    key=lambda n: self.mcts._avg_trajectory_entropy(n),
                    default=None,
                )
            )
            if chosen is not None:
                training_examples.extend(self._trajectory_to_examples(chosen, terminals))

        loss = self.policy.reinforce_update(
            training_examples,
            learning_rate=self.config.learning_rate,
            clip_low=self.config.clip_low,
            clip_high=self.config.clip_high,
            gradient_clip=self.config.gradient_clip,
        )
        return {
            "loss": loss,
            "training_examples": float(len(training_examples)),
            "mcts_problems": float(search_count),
            "new_cached_problems": float(correct_found),
            "replay_size": float(len(self.replay)),
        }

    def re_filter(self, hard_set: Sequence[ArithmeticProblem]) -> List[ArithmeticProblem]:
        return [
            p
            for p in hard_set
            if self.rollouts.pass1_at_k(p, self.config.pass_k) < self.config.hard_threshold
        ]

    def train(self, rounds: int = 5) -> List[Dict[str, float]]:
        hard_set = self.initial_hard_set()
        history: List[Dict[str, float]] = []
        for round_idx in range(rounds):
            stats = self.train_round(hard_set)
            stats["round"] = float(round_idx + 1)
            stats["hard_set_before"] = float(len(hard_set))
            hard_set = self.re_filter(hard_set)
            stats["hard_set_after"] = float(len(hard_set))
            history.append(stats)
            if not hard_set:
                break
        return history


# -----------------------------------------------------------------------------
# Demonstration
# -----------------------------------------------------------------------------


def make_demo_problems() -> List[ArithmeticProblem]:
    """Construct small verifiable tasks with mixed difficulty."""
    targets = [2, 4, 6, 8, 10, 12, -2, -4, 7, 9, 11, 13]
    return [
        ArithmeticProblem(problem_id=f"P{i:02d}", target=t, max_steps=5 if abs(t) <= 8 else 6)
        for i, t in enumerate(targets, start=1)
    ]


def evaluate(policy: TabularPolicy, problems: Sequence[ArithmeticProblem], k: int = 4) -> float:
    verifier = ToyVerifier()
    engine = RolloutEngine(policy, verifier, seed=123)
    if not problems:
        return 1.0
    return sum(engine.pass1_at_k(p, k) for p in problems) / len(problems)


def main() -> None:
    problems = make_demo_problems()

    config = DeepSearchConfig(
        max_depth=8,
        expansion_width=8,
        mcts_rollouts=24,
        lambda1=0.4,
        lambda2=0.0,
        lambda3=0.01,
        hard_threshold=0.25,
        pass_k=4,
        learning_rate=0.05,
        random_seed=7,
    )

    trainer = DeepSearchTrainer(problems, config)
    before = evaluate(trainer.policy, problems, k=4)
    history = trainer.train(rounds=5)
    after = evaluate(trainer.policy, problems, k=4)

    print("DeepSearch-style from-scratch prototype")
    print(f"Problems: {len(problems)}")
    print(f"Average Pass@1 over K=4 rollouts before training: {before:.3f}")
    for row in history:
        print(
            "Round {round:.0f}: hard {hard_set_before:.0f}->{hard_set_after:.0f}, "
            "MCTS={mcts_problems:.0f}, new_cache={new_cached_problems:.0f}, "
            "replay={replay_size:.0f}, examples={training_examples:.0f}, loss={loss:.4f}".format(**row)
        )
    print(f"Average Pass@1 over K=4 rollouts after training:  {after:.3f}")
    print(f"Final replay-buffer size: {len(trainer.replay)}")


if __name__ == "__main__":
    main()
