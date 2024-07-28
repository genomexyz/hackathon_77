import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

# Load dataset
df = pd.read_csv('rain_prediction_data.csv')

# Encode 'rain' column to numerical values
label_encoder = LabelEncoder()
df['rain_encoded'] = label_encoder.fit_transform(df['rain'])

# Discretize continuous features into bins
n_bins = 5
est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
discretized_features = est.fit_transform(df[['temperature', 'wind_speed', 'humidity']])

# Add discretized features to dataframe
df['temp_bin'] = discretized_features[:, 0]
df['wind_speed_bin'] = discretized_features[:, 1]
df['humidity_bin'] = discretized_features[:, 2]

# Calculate emission probabilities
def calculate_emission_probabilities(df, n_bins, state):
    probs = []
    for feature in ['temp_bin', 'wind_speed_bin', 'humidity_bin']:
        counts = df[df['rain_encoded'] == state][feature].value_counts(normalize=True).sort_index()
        prob = counts.reindex(range(n_bins), fill_value=0).values
        probs.append(prob)
        print('cek prob', feature, prob)
    return probs

# Calculate emission probabilities for each state
emission_probs_no = calculate_emission_probabilities(df, n_bins, state=0)
emission_probs_yes = calculate_emission_probabilities(df, n_bins, state=1)

# Transition probabilities (example values)
trans_prob = {
    'no': {'no': 0.7, 'yes': 0.3},
    'yes': {'no': 0.4, 'yes': 0.6}
}

# Viterbi algorithm
def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = start_prob[state] * emit_prob[state][0][int(obs[0][0])] * emit_prob[state][1][int(obs[0][1])] * emit_prob[state][2][int(obs[0][2])]
        path[state] = [state]

    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, prev_state) = max(
                (V[t-1][prev] * trans_prob[prev][state] * emit_prob[state][0][int(obs[t][0])] * emit_prob[state][1][int(obs[t][1])] * emit_prob[state][2][int(obs[t][2])], prev) for prev in states
            )
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    (prob, state) = max((V[-1][state], state) for state in states)
    return (prob, path[state])

# Observations (discretized features)
obs = discretized_features

# Initial state probabilities
start_prob = {'no': 0.5, 'yes': 0.5}

# Emission probabilities
emit_prob = {'no': emission_probs_no, 'yes': emission_probs_yes}

# States
states = ['no', 'yes']

# Run Viterbi algorithm
prob, state_sequence = viterbi(obs, states, start_prob, trans_prob, emit_prob)

print('Most probable sequence of states:', state_sequence)
