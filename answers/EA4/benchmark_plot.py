import pandas as pd
import matplotlib.pyplot as plt

# Load the benchmark results from the CSV file
data = pd.read_csv('benchmark_results.csv')

# Graph 1: Speedup vs. Number of Threads
plt.figure(figsize=(20, 15))  # Increase figure size for better readability
for size in data['ArraySize'].unique():
    subset = data[data['ArraySize'] == size]
    plt.plot(subset['NumThreads'], subset['StdSort'] / subset['MinMaxQuicksort'], label=f'MinMaxQuicksort Size {size}', linewidth=2)
    plt.plot(subset['NumThreads'], subset['StdSort'] / subset['GnuParallelSort'], label=f'GnuParallelSort Size {size}', linewidth=2)
plt.xlabel('Number of Threads', fontsize=15)
plt.ylabel('Speedup over std::sort', fontsize=15)
plt.title('Speedup vs. Number of Threads', fontsize=17)
plt.legend(fontsize=13)
plt.grid()
plt.savefig('speedup_vs_threads.png')

# Graph 2: Speedup vs. Array Size
plt.figure(figsize=(20, 15))  # Increase figure size for better readability
plt.xscale('log')
for threads in data['NumThreads'].unique():
    subset = data[data['NumThreads'] == threads]
    plt.plot(subset['ArraySize'], subset['StdSort'] / subset['MinMaxQuicksort'], label=f'MinMaxQuicksort Threads {threads}', linewidth=2)
    plt.plot(subset['ArraySize'], subset['StdSort'] / subset['GnuParallelSort'], label=f'GnuParallelSort Threads {threads}', linewidth=2)
plt.xlabel('Array Size', fontsize=15)
plt.ylabel('Speedup over std::sort', fontsize=15)
plt.title('Speedup vs. Array Size', fontsize=17)
plt.legend(fontsize=13)
plt.grid()
plt.savefig('speedup_vs_array_size.png')


# Filter the data to include only rows with 12 threads
subset_12_threads = data[data['NumThreads'] == 12]

# Graph 3: Speedup vs. Array Size for Threads = 12
plt.figure(figsize=(20, 15))  # Increase figure size for better readability
plt.xscale('log')
plt.plot(subset_12_threads['ArraySize'], subset_12_threads['StdSort'] / subset_12_threads['MinMaxQuicksort'], label='MinMaxQuicksort Threads 12', linewidth=2)
plt.plot(subset_12_threads['ArraySize'], subset_12_threads['StdSort'] / subset_12_threads['GnuParallelSort'], label='GnuParallelSort Threads 12', linewidth=2)

# Labels and Title
plt.xlabel('Array Size', fontsize=15)
plt.ylabel('Speedup over std::sort', fontsize=15)
plt.title('Speedup vs. Array Size for Threads = 12', fontsize=17)
plt.legend(fontsize=13)
plt.grid()

# Save and display the plot
plt.savefig('speedup_vs_array_size_threads_12.png')
#plt.show()
