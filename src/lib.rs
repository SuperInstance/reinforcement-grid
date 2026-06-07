//! # reinforcement-grid
//!
//! Grid world reinforcement learning environment with tabular Q-learning and SARSA.
//!
//! ## Example
//!
//! ```
//! use reinforcement_grid::{GridWorld, QLearning, QTable, Policy};
//!
//! let mut grid = GridWorld::new(4, 4);
//! grid.set_terminal(3, 3, 1.0);
//! grid.set_wall(1, 1);
//!
//! let mut qtable = QTable::new(4, 4);
//! let mut learner = QLearning::new(0.1, 0.9, 0.1);
//!
//! for _ in 0..1000 {
//!     let mut state = (0, 0);
//!     while !grid.is_terminal(state) {
//!         let action = learner.select_action(&qtable, state, &grid);
//!         let (next_state, reward) = grid.step(state, action);
//!         qtable.update(state, action, learner.delta(&qtable, state, action, reward, next_state));
//!         state = next_state;
//!     }
//! }
//! ```

/// Actions available in the grid world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    Up,
    Down,
    Left,
    Right,
}

impl Action {
    /// All four actions.
    pub fn all() -> &'static [Action; 4] {
        &[Action::Up, Action::Down, Action::Left, Action::Right]
    }

    /// Apply this action to a position, returning the new position.
    pub fn apply(&self, (r, c): (usize, usize)) -> (isize, isize) {
        match self {
            Action::Up => (r as isize - 1, c as isize),
            Action::Down => (r as isize + 1, c as isize),
            Action::Left => (r as isize, c as isize - 1),
            Action::Right => (r as isize, c as isize + 1),
        }
    }

    /// Number of possible actions.
    pub fn count() -> usize {
        4
    }
}

/// Cell type in the grid world.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Cell {
    /// Normal cell with a step reward (typically -1).
    Normal(f64),
    /// Wall — cannot be entered.
    Wall,
    /// Terminal cell with given reward. Episode ends here.
    Terminal(f64),
}

/// Grid world environment for reinforcement learning.
#[derive(Debug, Clone)]
pub struct GridWorld {
    rows: usize,
    cols: usize,
    cells: Vec<Vec<Cell>>,
    default_reward: f64,
}

impl GridWorld {
    /// Create a new grid world of the given size with normal cells.
    pub fn new(rows: usize, cols: usize) -> Self {
        assert!(rows > 0 && cols > 0, "grid must have positive dimensions");
        Self {
            rows,
            cols,
            cells: vec![vec![Cell::Normal(-1.0); cols]; rows],
            default_reward: -1.0,
        }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Set a cell to be a wall.
    pub fn set_wall(&mut self, r: usize, c: usize) {
        self.cells[r][c] = Cell::Wall;
    }

    /// Set a cell to be terminal with the given reward.
    pub fn set_terminal(&mut self, r: usize, c: usize, reward: f64) {
        self.cells[r][c] = Cell::Terminal(reward);
    }

    /// Set the reward for a normal cell.
    pub fn set_reward(&mut self, r: usize, c: usize, reward: f64) {
        self.cells[r][c] = Cell::Normal(reward);
    }

    /// Get the cell at a position.
    pub fn cell(&self, r: usize, c: usize) -> &Cell {
        &self.cells[r][c]
    }

    /// Check if a state is terminal.
    pub fn is_terminal(&self, (r, c): (usize, usize)) -> bool {
        matches!(self.cells[r][c], Cell::Terminal(_))
    }

    /// Check if a state is a wall.
    pub fn is_wall(&self, (r, c): (usize, usize)) -> bool {
        matches!(self.cells[r][c], Cell::Wall)
    }

    /// Check if a state is within bounds and not a wall.
    pub fn is_valid(&self, state: (isize, isize)) -> bool {
        let (r, c) = state;
        if r < 0 || c < 0 || r as usize >= self.rows || c as usize >= self.cols {
            return false;
        }
        !self.is_wall((r as usize, c as usize))
    }

    /// Get the reward for entering a cell.
    pub fn reward(&self, (r, c): (usize, usize)) -> f64 {
        match self.cells[r][c] {
            Cell::Normal(rw) => rw,
            Cell::Wall => 0.0,
            Cell::Terminal(rw) => rw,
        }
    }

    /// Take an action from a state. Returns (next_state, reward).
    /// If the action leads to a wall or out of bounds, the agent stays put.
    pub fn step(&self, state: (usize, usize), action: Action) -> ((usize, usize), f64) {
        let (nr, nc) = action.apply(state);
        if self.is_valid((nr, nc)) {
            let ns = (nr as usize, nc as usize);
            (ns, self.reward(ns))
        } else {
            (state, self.default_reward)
        }
    }

    /// Get all valid non-wall, non-terminal states.
    pub fn valid_states(&self) -> Vec<(usize, usize)> {
        let mut states = Vec::new();
        for r in 0..self.rows {
            for c in 0..self.cols {
                if !self.is_wall((r, c)) && !self.is_terminal((r, c)) {
                    states.push((r, c));
                }
            }
        }
        states
    }
}

/// State-action value table (Q-table) for tabular RL.
#[derive(Debug, Clone)]
pub struct QTable {
    /// Q values indexed as q[row][col][action_index].
    q: Vec<Vec<Vec<f64>>>,
    #[allow(dead_code)]
    rows: usize,
    #[allow(dead_code)]
    cols: usize,
}

impl QTable {
    /// Create a new Q-table initialized to zeros.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            q: vec![vec![vec![0.0; Action::count()]; cols]; rows],
            rows,
            cols,
        }
    }

    /// Get Q(s, a).
    pub fn get(&self, state: (usize, usize), action: Action) -> f64 {
        self.q[state.0][state.1][action as usize]
    }

    /// Set Q(s, a).
    pub fn set(&mut self, state: (usize, usize), action: Action, value: f64) {
        self.q[state.0][state.1][action as usize] = value;
    }

    /// Update Q(s, a) by adding delta.
    pub fn update(&mut self, state: (usize, usize), action: Action, delta: f64) {
        self.q[state.0][state.1][action as usize] += delta;
    }

    /// Get the best action for a state (greedy).
    pub fn best_action(&self, state: (usize, usize)) -> Action {
        let vals = &self.q[state.0][state.1];
        let best_idx = vals.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        match best_idx {
            0 => Action::Up,
            1 => Action::Down,
            2 => Action::Left,
            _ => Action::Right,
        }
    }

    /// Get the maximum Q value for a state.
    pub fn max_q(&self, state: (usize, usize)) -> f64 {
        self.q[state.0][state.1].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get all Q values for a state.
    pub fn values(&self, state: (usize, usize)) -> &[f64] {
        &self.q[state.0][state.1]
    }
}

/// Policy for action selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    /// ε-greedy: explore with probability ε, exploit otherwise.
    EpsilonGreedy(f64),
    /// Softmax (Boltzmann): select actions with probability proportional to exp(Q/τ).
    Softmax(f64),
}

impl Policy {
    /// Select an action according to this policy.
    /// Uses a simple deterministic RNG for reproducibility.
    pub fn select(&self, qtable: &QTable, state: (usize, usize), rng_value: f64) -> Action {
        match self {
            Policy::EpsilonGreedy(eps) => {
                if rng_value < *eps {
                    // Random action
                    let idx = (rng_value * Action::count() as f64 * 100.0) as usize % Action::count();
                    Action::all()[idx]
                } else {
                    qtable.best_action(state)
                }
            }
            Policy::Softmax(tau) => {
                let vals = qtable.values(state);
                let max_val = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = vals.iter().map(|v| ((v - max_val) / tau).exp()).collect();
                let sum: f64 = exps.iter().sum();
                let mut cumulative = 0.0;
                let threshold = rng_value * sum;
                for (i, e) in exps.iter().enumerate() {
                    cumulative += e;
                    if cumulative >= threshold {
                        return Action::all()[i];
                    }
                }
                Action::all()[3]
            }
        }
    }
}

/// Q-learning (off-policy TD control) agent.
#[derive(Debug, Clone)]
pub struct QLearning {
    learning_rate: f64,
    discount: f64,
    epsilon: f64,
    rng_state: u64,
}

impl QLearning {
    /// Create a new Q-learning agent with learning rate α, discount γ, and exploration ε.
    pub fn new(learning_rate: f64, discount: f64, epsilon: f64) -> Self {
        assert!(learning_rate > 0.0 && learning_rate <= 1.0);
        assert!((0.0..=1.0).contains(&discount));
        assert!((0.0..=1.0).contains(&epsilon));
        Self { learning_rate, discount, epsilon, rng_state: 42 }
    }

    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x & 0x7FFFFFFFFFFFFFFF) as f64 / u64::MAX as f64
    }

    /// Select an action using ε-greedy policy.
    pub fn select_action(&mut self, qtable: &QTable, state: (usize, usize), _grid: &GridWorld) -> Action {
        let policy = Policy::EpsilonGreedy(self.epsilon);
        policy.select(qtable, state, self.next_random())
    }

    /// Compute the Q-learning update delta: α * (r + γ * max_a' Q(s', a') - Q(s, a))
    pub fn delta(&self, qtable: &QTable, state: (usize, usize), action: Action, reward: f64, next_state: (usize, usize)) -> f64 {
        let current_q = qtable.get(state, action);
        let max_next_q = qtable.max_q(next_state);
        self.learning_rate * (reward + self.discount * max_next_q - current_q)
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Get the discount factor.
    pub fn discount(&self) -> f64 {
        self.discount
    }
}

/// SARSA (on-policy TD control) agent.
#[derive(Debug, Clone)]
pub struct Sarsa {
    learning_rate: f64,
    discount: f64,
    epsilon: f64,
    rng_state: u64,
}

impl Sarsa {
    /// Create a new SARSA agent.
    pub fn new(learning_rate: f64, discount: f64, epsilon: f64) -> Self {
        assert!(learning_rate > 0.0 && learning_rate <= 1.0);
        assert!((0.0..=1.0).contains(&discount));
        assert!((0.0..=1.0).contains(&epsilon));
        Self { learning_rate, discount, epsilon, rng_state: 99 }
    }

    fn next_random(&mut self) -> f64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        (x & 0x7FFFFFFFFFFFFFFF) as f64 / u64::MAX as f64
    }

    /// Select an action using ε-greedy policy.
    pub fn select_action(&mut self, qtable: &QTable, state: (usize, usize)) -> Action {
        let policy = Policy::EpsilonGreedy(self.epsilon);
        policy.select(qtable, state, self.next_random())
    }

    /// Compute the SARSA update delta: α * (r + γ * Q(s', a') - Q(s, a))
    pub fn delta(&self, qtable: &QTable, state: (usize, usize), action: Action, reward: f64, next_state: (usize, usize), next_action: Action) -> f64 {
        let current_q = qtable.get(state, action);
        let next_q = qtable.get(next_state, next_action);
        self.learning_rate * (reward + self.discount * next_q - current_q)
    }

    /// Get the learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Get the discount factor.
    pub fn discount(&self) -> f64 {
        self.discount
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_apply() {
        assert_eq!(Action::Up.apply((2, 2)), (1, 2));
        assert_eq!(Action::Down.apply((2, 2)), (3, 2));
        assert_eq!(Action::Left.apply((2, 2)), (2, 1));
        assert_eq!(Action::Right.apply((2, 2)), (2, 3));
    }

    #[test]
    fn test_grid_creation() {
        let grid = GridWorld::new(3, 4);
        assert_eq!(grid.rows(), 3);
        assert_eq!(grid.cols(), 4);
    }

    #[test]
    fn test_grid_wall() {
        let mut grid = GridWorld::new(3, 3);
        grid.set_wall(1, 1);
        assert!(grid.is_wall((1, 1)));
        assert!(!grid.is_wall((0, 0)));
    }

    #[test]
    fn test_grid_terminal() {
        let mut grid = GridWorld::new(3, 3);
        grid.set_terminal(2, 2, 10.0);
        assert!(grid.is_terminal((2, 2)));
        assert!((grid.reward((2, 2)) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_grid_step_valid() {
        let mut grid = GridWorld::new(4, 4);
        grid.set_terminal(3, 3, 10.0);
        let (ns, _reward) = grid.step((0, 0), Action::Right);
        assert_eq!(ns, (0, 1));
    }

    #[test]
    fn test_grid_step_wall() {
        let mut grid = GridWorld::new(3, 3);
        grid.set_wall(0, 1);
        let (ns, _) = grid.step((0, 0), Action::Right);
        assert_eq!(ns, (0, 0)); // Bounced back
    }

    #[test]
    fn test_grid_step_out_of_bounds() {
        let grid = GridWorld::new(3, 3);
        let (ns, _) = grid.step((0, 0), Action::Up);
        assert_eq!(ns, (0, 0)); // Stayed
    }

    #[test]
    fn test_qtable_get_set() {
        let mut qt = QTable::new(3, 3);
        assert_eq!(qt.get((1, 1), Action::Up), 0.0);
        qt.set((1, 1), Action::Up, 5.0);
        assert_eq!(qt.get((1, 1), Action::Up), 5.0);
    }

    #[test]
    fn test_qtable_best_action() {
        let mut qt = QTable::new(3, 3);
        qt.set((0, 0), Action::Up, 1.0);
        qt.set((0, 0), Action::Right, 5.0);
        qt.set((0, 0), Action::Down, 2.0);
        assert_eq!(qt.best_action((0, 0)), Action::Right);
    }

    #[test]
    fn test_qtable_max_q() {
        let mut qt = QTable::new(2, 2);
        qt.set((0, 0), Action::Down, 3.0);
        qt.set((0, 0), Action::Up, 7.0);
        assert!((qt.max_q((0, 0)) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_qtable_update() {
        let mut qt = QTable::new(2, 2);
        qt.update((0, 0), Action::Up, 2.5);
        assert!((qt.get((0, 0), Action::Up) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_policy_epsilon_greedy() {
        let mut qt = QTable::new(2, 2);
        qt.set((0, 0), Action::Right, 10.0);
        let policy = Policy::EpsilonGreedy(0.0); // Pure exploit
        assert_eq!(policy.select(&qt, (0, 0), 0.5), Action::Right);
    }

    #[test]
    fn test_policy_softmax() {
        let mut qt = QTable::new(2, 2);
        qt.set((0, 0), Action::Right, 10.0);
        let policy = Policy::Softmax(0.1); // Low temperature → exploit
        // With very low tau, should almost always pick the best action
        let action = policy.select(&qt, (0, 0), 0.5);
        assert_eq!(action, Action::Right);
    }

    #[test]
    fn test_qlearning_delta() {
        let mut qt = QTable::new(3, 3);
        qt.set((0, 0), Action::Right, 0.0);
        qt.set((0, 1), Action::Right, 0.0);
        let learner = QLearning::new(0.1, 0.9, 0.1);
        let delta = learner.delta(&qt, (0, 0), Action::Right, -1.0, (0, 1));
        assert!(delta != 0.0);
    }

    #[test]
    fn test_sarsa_delta() {
        let mut qt = QTable::new(3, 3);
        qt.set((0, 0), Action::Right, 0.0);
        qt.set((0, 1), Action::Down, 0.0);
        let learner = Sarsa::new(0.1, 0.9, 0.1);
        let delta = learner.delta(&qt, (0, 0), Action::Right, -1.0, (0, 1), Action::Down);
        assert!(delta != 0.0);
    }

    #[test]
    fn test_grid_valid_states() {
        let mut grid = GridWorld::new(3, 3);
        grid.set_wall(1, 1);
        grid.set_terminal(2, 2, 10.0);
        let states = grid.valid_states();
        assert_eq!(states.len(), 7); // 9 - 1 wall - 1 terminal
        assert!(!states.contains(&(1, 1)));
        assert!(!states.contains(&(2, 2)));
    }
}
