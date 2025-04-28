import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Cursor

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

import json

# Dictionary to hold closest (k, t) pairs for each threshold
closest_points_dict = {}

for threshold in thresholds:
    # Load the datasets
    df1 = pd.read_csv(f'../../data/cascading/outputs/cascade_results/val_threshold{threshold}.csv')
    df2 = pd.read_csv(f'../../data/cascading/outputs/pareto_points/pareto_threshold{threshold}.csv')
    
    # Create a scatter plot for the first dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # First plot all light blue dots
    for _, row in df1.iterrows():
        non_negative_count = sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']])
        if non_negative_count > 1:
            ax.scatter(row['cost'], row['accuracy'], color='lightgreen', zorder=1)
        else:
            ax.scatter(row['cost'], row['accuracy'], color='lightblue', zorder=1)
    
    # Draw grid lines
    plt.grid(which='both', color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Store all points data
    all_points = []
    
    # Add cascaded non-Pareto points
    for _, row in df1.iterrows():
        all_points.append({
            'cost': row['cost'],
            'accuracy': row['accuracy'],
            'k1': row['k1'],
            'k2': row['k2'],
            'k3': row['k3'],
            't1': row['t1'],
            't2': row['t2'],
            't3': row['t3'],
            'type': 'cascaded' if sum(n >= 0 for n in [row['k1'], row['k2'], row['k3']]) > 1 else 'standalone',
            'pareto': False
        })
    
    # Store scatter points from df2
    pareto_points = []
    green_points = []
    mediumblue_points = []
    
    # Overlay the second dataset (green dots and purple crosses)
    for idx, row in df2.iterrows():
        is_singular = row['Singular'] == 1
        color = 'mediumblue' if is_singular else 'green'
        size = 40 if color == 'mediumblue' else 20
        
        scatter = ax.scatter(row['cost'], row['accuracy'], color=color, marker='o', s=size, zorder=3, picker=True)
        
        point_data = {
            'cost': row['cost'],
            'accuracy': row['accuracy'],
            'k1': row['k1'],
            'k2': row['k2'],
            'k3': row['k3'],
            't1': row['t1'],
            't2': row['t2'],
            't3': row['t3'],
            'type': 'standalone' if is_singular else 'cascaded',
            'pareto': True,
            'scatter': scatter
        }
        
        pareto_points.append(point_data)
        all_points.append(point_data)
        
        if color == 'green':
            green_points.append(point_data)
        else:
            mediumblue_points.append(point_data)
    
    # Define the theoretical best point: cost=0, accuracy=max possible
    # Find the maximum accuracy value from all points
    max_accuracy = max([p['accuracy'] for p in all_points])
    theoretical_best = {'cost': 0, 'accuracy': max_accuracy, 'k1': 'N/A', 'k2': 'N/A', 'k3': 'N/A', 't1': 'N/A', 't2': 'N/A', 't3': 'N/A'}
    
    # Mark the theoretical best point with a star (no annotation)
    ax.scatter(theoretical_best['cost'], theoretical_best['accuracy'], color='red', marker='*', s=200, zorder=4)
    
    # Normalize cost and accuracy for all points to calculate weighted distance
    costs = [p['cost'] for p in all_points]
    accuracies = [p['accuracy'] for p in all_points]
    
    cost_min, cost_max = min(costs), max(costs)
    acc_min, acc_max = min(accuracies), max(accuracies)
    
    # Calculate weighted distance from each point to the theoretical best point
    weight_cost = 0.5
    weight_accuracy = 1 - weight_cost
    
    # Function to calculate normalized weighted distance
    def calculate_distance(point, best_point):
        normalized_cost_diff = 0
        if cost_max > cost_min:  # Avoid division by zero
            normalized_cost_diff = (point['cost'] - best_point['cost']) / (cost_max - cost_min)
            
        normalized_acc_diff = 0
        if acc_max > acc_min:  # Avoid division by zero
            normalized_acc_diff = (best_point['accuracy'] - point['accuracy']) / (acc_max - acc_min)
            
        weighted_dist = weight_cost * normalized_cost_diff**2 + weight_accuracy * normalized_acc_diff**2
        return np.sqrt(weighted_dist)
    
    # Find closest green (cascaded) point
    if green_points:
        closest_green = min(green_points, key=lambda p: calculate_distance(p, theoretical_best))
        ax.annotate(f"Closest cascaded:\nk1={int(closest_green['k1'])}, k2={int(closest_green['k2'])}, k3={int(closest_green['k3'])}\nt1={int(closest_green['t1'])}, t2={int(closest_green['t2'])}, t3={int(closest_green['t3'])}",
                   xy=(closest_green['cost'], closest_green['accuracy']),
                   xytext=(80, -40), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='green'))
        
        # Draw dashed line from closest green to theoretical best
        ax.plot([closest_green['cost'], theoretical_best['cost']], 
                [closest_green['accuracy'], theoretical_best['accuracy']], 
                'g--', linewidth=1.5)
    
    # Find closest mediumblue (standalone) point
    if mediumblue_points:
        closest_blue = min(mediumblue_points, key=lambda p: calculate_distance(p, theoretical_best))
        # Modified position to place annotation below the point (negative y offset)
        ax.annotate(f"Closest standalone:\nk1={int(closest_blue['k1'])}, k2={int(closest_blue['k2'])}, k3={int(closest_blue['k3'])}\nt1={int(closest_blue['t1'])}, t2={int(closest_blue['t2'])}, t3={int(closest_blue['t3'])}",
                   xy=(closest_blue['cost'], closest_blue['accuracy']),
                   xytext=(80, -110), textcoords='offset points',  # Changed from +30 to -40
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="mediumblue", alpha=0.8),
                   arrowprops=dict(arrowstyle='->', color='mediumblue'))
        
        # Draw dashed line from closest blue to theoretical best
        ax.plot([closest_blue['cost'], theoretical_best['cost']], 
                [closest_blue['accuracy'], theoretical_best['accuracy']], 
                'b--', linewidth=1.5)
        
    closest_green_tuple = (
        [int(closest_green['k1']), int(closest_green['k2']), int(closest_green['k3'])],
        [int(closest_green['t1']), int(closest_green['t2']), int(closest_green['t3'])]
    ) if green_points else (None, None)

    closest_blue_tuple = (
        [int(closest_blue['k1']), int(closest_blue['k2']), int(closest_blue['k3'])],
        [int(closest_blue['t1']), int(closest_blue['t2']), int(closest_blue['t3'])]
    ) if mediumblue_points else (None, None)

    # Save into the dictionary
    closest_points_dict[str(threshold)] = {
        'closest_standalone': closest_blue_tuple,
        'closest_cascaded': closest_green_tuple
    }
    
    # Setup the hover annotation
    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                      arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            for point in pareto_points:
                cont, ind = point['scatter'].contains(event)
                if cont:
                    pos = point['scatter'].get_offsets()[0]
                    annot.xy = pos
                    text = f"k1={int(point['k1'])}, k2={int(point['k2'])}, k3={int(point['k3'])}\nt1={int(point['t1'])}, t2={int(point['t2'])}, t3={int(point['t3'])}"
                    annot.set_text(text)
                    annot.get_bbox_patch().set_alpha(0.8)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                    return
            
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()
    
    # Connect hover event
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    # Add legends
    plt.scatter([], [], color='lightgreen', label='Cascaded non-Pareto-optimal')
    plt.scatter([], [], color='lightblue', label='Standalone non-Pareto-optimal')
    plt.scatter([], [], color='green', label='Cascaded Pareto-optimal')
    plt.scatter([], [], color='mediumblue', label='Standalone Pareto-optimal')
    plt.scatter([], [], color='red', marker='*', s=100, label='Theoretical best point')
    
    plt.xlabel('Cost per 1k queries ($)', fontsize=13)
    plt.ylabel('Accuracy (%)', fontsize=13)
    plt.title(f'WizardCoder family performance on the validation set for threshold={threshold}', fontsize=14)
    
    # Shift the legend to lower right
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

# Save to JSON
with open('closest_points.json', 'w') as f:
    json.dump(closest_points_dict, f, indent=4)
