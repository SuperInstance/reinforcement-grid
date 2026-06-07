# reinforcement-grid

A Rust library providing a **grid world reinforcement learning environment** with tabular Q-learning and SARSA implementations.

[![crates.io](https://img.shields.io/crates/v/reinforcement-grid.svg)](https://crates.io/crates/reinforcement-grid)
[![Documentation](https://docs.rs/reinforcement-grid/badge.svg)](https://docs.rs/reinforcement-grid)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Overview

Grid worlds are the "Hello World" of reinforcement learning. This library provides a complete, self-contained environment for experimenting with tabular RL algorithms:

- **GridWorld** — Configurable 2D environment with walls, terminals, and rewards
- **QTable** — State-action value storage
- **Policy** — ε-greedy and softmax action selection
- **QLearning** — Off-policy TD control
- **SARSA** — On-policy TD control

No external dependencies. Pure Rust. Perfect for learning, teaching, and benchmarking RL algorithms.

## Installation

```toml
[dependencies]
reinforcement-grid = "0.1.0"
```

## Quick Start

### Creating an Environment

```rust
use reinforcement_grid::{GridWorld, Action};

let mut grid = GridWorld::new(4, 4);
grid.set_terminal(3, 3, 10.0);  // Goal with reward +10
grid.set_wall(1, 1);             // Obstacle
grid.set_wall(2, 2);             // Another obstacle

// Take a step
let (next_state, reward) = grid.step((0, 0), Action::Right);
println!("Moved to {:?}, got reward {}", next_state, reward);
```

### Training with Q-Learning

```rust
use reinforcement_grid::{GridWorld, QTable, QLearning};

let mut grid = GridWorld::new(4, 4);
grid.set_terminal(3, 3, 10.0);

let mut qtable = QTable::new(4, 4);
let mut learner = QLearning::new(0.1, 0.95, 0.1);

// Train for 1000 episodes
for _ in 0..1000 {
    let mut state = (0, 0);
    while !grid.is_terminal(state) {
        let action = learner.select_action(&qtable, state, &grid);
        let (next_state, reward) = grid.step(state, action);
        let delta = learner.delta(&qtable, state, action, reward, next_state);
        qtable.update(state, action, delta);
        state = next_state;
    }
}

// Get the learned policy
let best = qtable.best_action((0, 0));
println!("Best action from (0,0): {:?}", best);
```

### Training with SARSA

```rust
use reinforcement_grid::{GridWorld, QTable, Sarsa};

let mut grid = GridWorld::new(4, 4);
grid.set_terminal(3, 3, 10.0);

let mut qtable = QTable::new(4, 4);
let mut learner = Sarsa::new(0.1, 0.95, 0.1);

for _ in 0..1000 {
    let mut state = (0, 0);
    let mut action = learner.select_action(&qtable, state);
    while !grid.is_terminal(state) {
        let (next_state, reward) = grid.step(state, action);
        let next_action = learner.select_action(&qtable, next_state);
        let delta = learner.delta(&qtable, state, action, reward, next_state, next_action);
        qtable.update(state, action, delta);
        state = next_state;
        action = next_action;
    }
}
```

## Core Concepts

### Grid World

A 2D grid where each cell can be:
- **Normal** — Traversable, with an associated step reward (typically -1)
- **Wall** — Impassable barrier
- **Terminal** — Ends the episode with a reward

### Q-Learning (Off-Policy)

Updates Q-values using the maximum future reward:

```
Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
```

Q-learning learns the optimal policy regardless of the agent's behavior, making it **off-policy**.

### SARSA (On-Policy)

Updates Q-values using the action actually taken:

```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
```

SARSA learns the value of the policy being followed, making it **on-policy**. This tends to produce safer policies near hazards.

### Policies

| Policy | Description | When to Use |
|--------|-------------|-------------|
| ε-Greedy | Explore randomly with probability ε | Default choice, easy to tune |
| Softmax | Boltzmann exploration with temperature τ | Smoother exploration, temperature tuning |

## API Reference

| Type | Description |
|------|-------------|
| `Action` | Up, Down, Left, Right |
| `Cell` | Normal, Wall, Terminal |
| `GridWorld` | 2D RL environment |
| `QTable` | State-action value storage |
| `Policy` | ε-greedy and softmax selection |
| `QLearning` | Off-policy TD(0) control |
| `Sarsa` | On-policy TD(0) control |

## Examples

### Cliff Walking

```rust
let mut grid = GridWorld::new(4, 12);
// Bottom row (except start and goal) is a cliff
for c in 1..11 {
    grid.set_terminal(3, c, -100.0);
}
grid.set_terminal(3, 11, 0.0); // Goal
```

SARSA learns to walk along the top (safer), while Q-learning learns the optimal but risky path along the cliff edge.

## Testing

```bash
cargo test
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open an issue or PR at [GitHub](https://github.com/SuperInstance/reinforcement-grid).
