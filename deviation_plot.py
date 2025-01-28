import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with actual data)
predictions = np.array(df['CCS_pred'])
true_values= np.array(df['CCS_actual'])
print(len(true_values))

# Compute deviations
deviation = [(b - a) / (a + 1e-8) * 100 for a, b in zip(true_values, predictions)]

# Round deviations
deviation_percentage_rounded = [round(d, 1) for d in deviation]

deviations_above_20 = sum(1 for d in deviation_percentage_rounded if abs(d) > 20)

# Calculate the percentage of data with deviation higher than 50%
percentage_above_20 = (deviations_above_20 / len(deviation_percentage_rounded)) * 100

print(f"Percentage of data with deviation > 20%: {percentage_above_50:.2f}%")

max_deviation_idx = np.argmax(deviation_percentage_rounded)
min_deviation_idx = np.argmin(deviation_percentage_rounded)
# Print summary
print(f"Max Deviation: {deviation_percentage_rounded[max_deviation_idx]}%")
print(f"True Value (Max Deviation): {true_values[max_deviation_idx]}")
print(f"Predicted Value (Max Deviation): {predictions[max_deviation_idx]}")
print(f"Index of Max Deviation: {max_deviation_idx}")

print(f"Min Deviation: {deviation_percentage_rounded[min_deviation_idx]}%")
print(f"True Value (Min Deviation): {true_values[min_deviation_idx]}")
print(f"Predicted Value (Min Deviation): {predictions[min_deviation_idx]}")
print(f"Index of Min Deviation: {min_deviation_idx}")


sorted_deviations = sorted(deviation_percentage_rounded)

# Calculate the index for the 5% threshold
lower_5_percent_index = int(len(sorted_deviations) * 0.01)
upper_5_percent_index = int(len(sorted_deviations) * 0.99)

# Filter out the top and bottom 5% of the data
filtered_deviations = sorted_deviations[lower_5_percent_index:upper_5_percent_index]
median_error = np.median(sorted_deviations)
# Plot the histogram with the filtered data
plt.hist(filtered_deviations, bins=np.arange(-80, 80, 2), edgecolor='black')

plt.axvline(median_error, color='red', linestyle='dashed', linewidth=2, label=f'Median Error: {median_error:.1f}%')

# Annotate the median value with the text "Median Error: {median_error}"
plt.text(median_error + 2, plt.ylim()[1] * 0.9, f'Median rel. error: {median_error:.1f}%', color='black', fontsize=10)

ax = plt.gca()  # Get the current axes
ax.set_facecolor("white")
# Labels and title
plt.xlabel('Peptide CCS Prediction Deviation (%)')
plt.ylabel('Frequency')

# Show plot
plt.show()
