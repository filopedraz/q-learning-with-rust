// Import the required modules
use std::collections::HashMap;
use rand::Rng;

// Define the constants
const STATES: usize = 5;
const ACTIONS: usize = 2;
const ALPHA: f64 = 0.1;
const GAMMA: f64 = 0.9;
const EPISODES: usize = 1000;

// Define the rewards matrix
const R: [[i32; ACTIONS]; STATES] = [
    [0, -10],
    [0, 10],
    [0, -50],
    [0, 50],
    [0, 100],
];

// Define the transitions matrix
const TRANSITIONS: [[usize; ACTIONS]; STATES] = [
    [1, 2],
    [3, 4],
    [0, 1],
    [2, 4],
    [2, 3],
];

fn main() {
    // Initialize the Q-values
    let mut q_values = HashMap::new();
    for s in 0..STATES {
        for a in 0..ACTIONS {
            q_values.insert((s, a), 0.0);
        }
    }

    // Q-Learning algorithm
    for _ in 0..EPISODES {
        let mut state = 0;
        while state != 4 {
            let action = select_action(state, &q_values);
            let next_state = TRANSITIONS[state][action];
            let reward = R[state][action] as f64;
            let max_q_next = (0..ACTIONS)
                .map(|a| q_values[&(next_state, a)])
                .fold(f64::NEG_INFINITY, f64::max);

            let q = q_values[&(state, action)];
            q_values.insert((state, action), q + ALPHA * (reward + GAMMA * max_q_next - q));

            state = next_state;
        }
    }

    // Display the learned Q-values
    for s in 0..STATES {
        for a in 0..ACTIONS {
            print!("{:.2} ", q_values[&(s, a)]);
        }
        println!();
    }
}

// Define the function for action selection
fn select_action(state: usize, q_values: &HashMap<(usize, usize), f64>) -> usize {
    let mut rng = rand::thread_rng();
    if rng.gen_bool(1.0 / (1.0 + f64::from((EPISODES / 10) as u32))) {
        rng.gen_range(0..ACTIONS)
    } else {
        (0..ACTIONS).max_by(|a, b| q_values[&(state, *a)].partial_cmp(&q_values[&(state, *b)]).unwrap()).unwrap()
    }
}
