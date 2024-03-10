import numpy as np
import matplotlib.pyplot as plt

N = 10  # Length of the binary vector
num_samples = 10000  # Number of samples to generate
independence_levels = np.linspace(0, 1, 11)  # From fully dependent (0) to fully independent (1)
unique_states = []

for independence in independence_levels:
    samples = []
    for _ in range(num_samples):
        vector = [np.random.choice([0, 1])]
        for i in range(1, N):
            if np.random.rand() < independence:
                vector.append(np.random.choice([0, 1]))  # Independent choice
            else:
                vector.append(vector[-1])  # Dependent on the previous position
        samples.append(tuple(vector))
    unique_states.append(len(set(samples)))

plt.plot(independence_levels, unique_states)
plt.xlabel('Degree of Independence')
plt.ylabel('Number of Unique States')
plt.title('Effect of Independence on Encoding Capacity')
plt.show()
