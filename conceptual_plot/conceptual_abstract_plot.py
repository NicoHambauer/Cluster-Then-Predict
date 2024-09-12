from matplotlib import pyplot as plt
import numpy as np

# Data for the plot
age = np.linspace(0, 100, 500)
global_model = np.maximum(100 - 0.5 * age, 0)
cluster_then_predict = np.piecewise(age, [age < 60, age >= 60], [lambda x: 100 - 2 * x, lambda x: (x - 60) * 1.5])

# Adding retiring age line
retiring_age = 65

# Adjusting the data for the cluster then predict model to ensure a smooth transition at the kink point
def cluster_then_predict_model(x):
    if x < 60:
        return 100 - 2 * x
    else:
        return 100 - 2 * 60 + (x - 60) * 1.5

# Creating the adjusted data
cluster_then_predict_adjusted = np.array([cluster_then_predict_model(x) for x in age])

# Plotting
plt.figure(figsize=(10, 6))

# Plotting the lines
plt.plot(age, global_model, label='Global Model', color='orange', linewidth=2)
plt.plot(age, cluster_then_predict_adjusted, label='Cluster-then-Predict', color='blue', linewidth=2)

# Adding retiring age line
plt.axvline(x=retiring_age, color='gray', linestyle='--')
plt.text(retiring_age + 2, 90, 'Retiring Age', color='gray', rotation=0, verticalalignment='top')

# Marking clusters
plt.text(20, -18, 'Cluster 1', color='black')
plt.text(80, -18, 'Cluster 2', color='black')

# Adding labels (removing title as it will be added in LaTeX)
plt.xlabel('Age')
plt.ylabel('Outdoor Activity')
# plt.title('Conceptual Plot: Cluster-then-Predict Model vs Global Model')  # Title removed

plt.legend()

# Customize x-axis to show only rough marks
plt.xticks(ticks=np.arange(0, 101, 10))

# Show plot
plt.grid(False)
plt.savefig('conceptual_abstract_plot.png')
plt.show()
