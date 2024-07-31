import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import filedialog
from tkinter import scrolledtext

# Create the GUI window
window = tk.Tk()

# Function to handle button click for selecting the input file
def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)

# Function to handle button click for clustering
def cluster_data():
    # Get the file path from the entry field
    file_path = file_entry.get()

    # Load the dataset
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data = data.dropna()

    # Get the percentage of data to be read
    percentage = float(percentage_entry.get())

    # Calculate the number of rows to be read
    num_rows = int(len(data) * (percentage / 100))

    # Read the specified number of rows from the dataset
    data = data.head(num_rows)

    # Extract the relevant columns for clustering
    X_categorical = data['status_type'].values.reshape(-1, 1)
    X_numeric = data.select_dtypes(include=['float64', 'int64']).values

    # Perform one-hot encoding on the categorical column
    encoder = OneHotEncoder()
    X_categorical_encoded = encoder.fit_transform(X_categorical).toarray()

    # Concatenate the encoded categorical column with the numeric columns
    X_encoded = np.concatenate((X_categorical_encoded, X_numeric), axis=1)

    # Normalize the data
    X_normalized = (X_encoded - X_encoded.min(axis=0)) / (X_encoded.max(axis=0) - X_encoded.min(axis=0))

    # Get the number of clusters from the user
    k = int(cluster_entry.get())

    # Randomly select initial centroids from the dataset
    np.random.seed(42)
    centroid_indices = np.random.choice(X_normalized.shape[0], size=k, replace=False)
    centroids = X_normalized[centroid_indices]

    # Perform K-means clustering
    max_iterations = 100
    for _ in range(max_iterations):
        # Assign each data point to the closest centroid
        distances = np.linalg.norm(X_normalized[:, np.newaxis] - centroids, axis=-1)
        labels = np.argmin(distances, axis=-1)

        # Update the centroids
        new_centroids = np.array([X_normalized[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    # Add the cluster labels to the original dataset
    data['Cluster'] = labels

    # Calculate the Manhattan distance for each data point from its centroid
    distances = np.abs(X_encoded - centroids[labels])

    # Calculate the outlier score as the sum of distances across all features
    outlier_scores = np.sum(distances, axis=1)

    # Find the indices of the top outliers
    top_outliers_indices = np.argsort(outlier_scores)[::-1]

    # Get the outlier records
    outlier_records = data.iloc[top_outliers_indices]

    # Clear the text widget
    output_text.delete(1.0, tk.END)

    # Print the clusters and outlier records
    for cluster in range(k):
        cluster_records = data[data['Cluster'] == cluster]
        output_text.insert(tk.END, f"Cluster {cluster} records:\n")
        output_text.insert(tk.END, f"{cluster_records}\n\n")

    output_text.insert(tk.END, "Outlier records:\n")
    output_text.insert(tk.END, f"{outlier_records}")

# Create and configure GUI elements
file_label = tk.Label(window, text="Input File:")
file_label.pack()
file_entry = tk.Entry(window)
file_entry.pack()
file_button = tk.Button(window, text="Select File", command=select_file)
file_button.pack()

percentage_label = tk.Label(window, text="Percentage of Data to Read:")
percentage_label.pack()
percentage_entry = tk.Entry(window)
percentage_entry.pack()

cluster_label = tk.Label(window, text="Number of Clusters (k):")
cluster_label.pack()
cluster_entry = tk.Entry(window)
cluster_entry.pack()

cluster_button = tk.Button(window, text="Cluster Data", command=cluster_data)
cluster_button.pack()

output_label = tk.Label(window, text="Output:")
output_label.pack()
output_text = scrolledtext.ScrolledText(window, width=100, height=70)
output_text.pack()

# Start the GUI event loop
window.mainloop()