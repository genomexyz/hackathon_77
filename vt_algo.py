import numpy as np

# Example HMM parameters
states = ['Sunny', 'Rainy']
observations = ['walk', 'shop', 'clean']

start_prob = {'Sunny': 0.6, 'Rainy': 0.4}

trans_prob = {
    'Sunny': {'Sunny': 0.7, 'Rainy': 0.3},
    'Rainy': {'Sunny': 0.4, 'Rainy': 0.6}
}

emit_prob = {
    'Sunny': {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
    'Rainy': {'walk': 0.6, 'shop': 0.3, 'clean': 0.1}
}

# Observations sequence
obs_seq = ['walk', 'shop', 'clean']

# Function to run the Viterbi algorithm
def viterbi(obs_seq, states, start_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}

    # Initialize the Viterbi table
    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state][obs_seq[0]]
        path[state] = [state]

    # Run the Viterbi algorithm
    for t in range(1, len(obs_seq)):
        V.append({})
        new_path = {}

        for state in states:
            max_prob, prev_state = max(
                (V[t-1][prev_state] * trans_prob[prev_state][state] * emit_prob[state][obs_seq[t]], prev_state)
                for prev_state in states
            )
            V[t][state] = max_prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    # Termination step
    max_prob, final_state = max((V[-1][state], state) for state in states)
    return max_prob, path[final_state], V

# Run Viterbi for the given observations
max_prob, state_sequence, V = viterbi(obs_seq, states, start_prob, trans_prob, emit_prob)
print(f'Most probable sequence: {state_sequence} with probability {max_prob}')

# Predicting three time steps ahead
future_steps = 3
current_prob = {state: V[-1][state] for state in states}

for _ in range(future_steps):
    next_prob = {}
    for next_state in states:
        next_prob[next_state] = sum(
            current_prob[state] * trans_prob[state][next_state]
            for state in states
        )
    current_prob = next_prob

print(f'Probabilities of states {future_steps} steps ahead: {current_prob}')
