import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('rain_prediction_data.csv')

# Encode 'rain' column to numerical values
label_encoder = LabelEncoder()
df['rain_encoded'] = label_encoder.fit_transform(df['rain'])

# Define states and observations
states = ['no', 'yes']
observations = df[['temperature', 'wind_speed', 'humidity']].values

# Initial state probabilities
start_prob = {'no': 0.5, 'yes': 0.5}

# Transition probabilities
trans_prob = {
    'no': {'no': 0.7, 'yes': 0.3},
    'yes': {'no': 0.4, 'yes': 0.6}
}

# Emission probabilities (dummy values for example purposes)
# In practice, these should be derived from the data
emit_prob = {
    'no': [0.5, 0.2, 0.3],  # Probabilities of observing temperature, wind_speed, humidity when no rain
    'yes': [0.2, 0.3, 0.5]  # Probabilities of observing temperature, wind_speed, humidity when rain
}

def viterbi(obs, states, start_prob, trans_prob, emit_prob):
    V = [{}]
    path = {}

    for state in states:
        V[0][state] = start_prob[state] * np.prod(emit_prob[state])
        path[state] = [state]

    for t in range(1, len(obs)):
        V.append({})
        new_path = {}

        for state in states:
            (prob, prev_state) = max(
                (V[t-1][prev] * trans_prob[prev][state] * np.prod(emit_prob[state]), prev) for prev in states
            )
            V[t][state] = prob
            new_path[state] = path[prev_state] + [state]

        path = new_path

    (prob, state) = max((V[-1][state], state) for state in states)
    return (prob, path[state])

# Run Viterbi algorithm
obs = list(df['rain_encoded'])
prob, state_sequence = viterbi(obs, states, start_prob, trans_prob, emit_prob)

print('Most probable sequence of states:', state_sequence)
