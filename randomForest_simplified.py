import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/Users/fabianschweisthal/Documents/Uni/2. Semester/PBL/Dataset/ccs/ccs_data.csv'
data = pd.read_csv(file_path, sep=',')

# Select relevant columns and rename them for consistency
data = data.rename(columns={
    'NumRotatableBonds': 'num_rotatable_bonds',
    'FractionSP3': 'fraction_sp3',
    'Length': 'molecule_length'
})
data = data[['Sequence', 'Charge', 'CCS', 'num_rotatable_bonds', 'fraction_sp3', 'molecule_length']]

# Define the 20 standard amino acids and include non-standard ones
amino_acids = 'ACDEFGHIKLMNPQRSTVWYU'
aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}

def one_hot_encode(seq, max_length):
    # Pad the sequence to the maximum length with 'X' (unknown amino acid)
    padded_seq = seq.ljust(max_length, 'X')
    int_encoded = [aa_to_int.get(aa, len(amino_acids)) for aa in padded_seq]  # Use a default index for unknown amino acids
    one_hot_encoded = np.zeros((max_length, len(amino_acids) + 1))  # +1 for the unknown amino acid
    for i, value in enumerate(int_encoded):
        one_hot_encoded[i, value] = 1
    return one_hot_encoded.flatten()

# Determine the maximum sequence length
max_seq_length = max(data['Sequence'].apply(len))

# Encode the sequences
one_hot_encoded_sequences = np.array([one_hot_encode(seq, max_seq_length) for seq in data['Sequence']])

# Initialize the label encoder for charge states
label_encoder = LabelEncoder()
encoded_charge_states = label_encoder.fit_transform(data['Charge'].tolist())

# Calculate additional features
sequence_lengths = data['Sequence'].apply(len).values.reshape(-1, 1)

def calculate_composition(seq, amino_acids):
    count = Counter(seq)
    return [count[aa] for aa in amino_acids]

amino_acid_composition = np.array([calculate_composition(seq, amino_acids) for seq in data['Sequence']])

# Combine all features
encoded_features = np.hstack((
    one_hot_encoded_sequences,
    encoded_charge_states.reshape(-1, 1),
    sequence_lengths,
    amino_acid_composition,
    data[['num_rotatable_bonds', 'fraction_sp3', 'molecule_length']].values
))

# Define the target variable
ccs_values = data['CCS'].values

# Normalize the CCS values
scaler = MinMaxScaler()
normalized_ccs_values = scaler.fit_transform(ccs_values.reshape(-1, 1)).flatten()

# Train-Test Split with random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(encoded_features, normalized_ccs_values, test_size=0.2, random_state=42)

# Train the Random Forest Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")
