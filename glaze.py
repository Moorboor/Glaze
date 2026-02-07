import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def psi_function(L_prev, H):
    """
    Calculates the prior expectation (Psi) for the Glaze model.

    This function distorts the previous belief based on the hazard rate H,
    representing the probability that the state has switched since the last trial.
    See Glaze et al. (2015) Eq 2.

    Args:
        L_prev (float): The belief (log-likelihood ratio) from the previous time step.
        H (float): The hazard rate (probability of state switch).

    Returns:
        float: The prior expectation for the current time step.
    """
    # Prevent numerical overflow for very large L
    # If L is huge, the prior is just the bound
    if abs(L_prev) > 100:
        return np.sign(L_prev) * np.log((1 - H) / H)

    term1 = (1 - H) / H
    term_pos = term1 + np.exp(-L_prev)
    term_neg = term1 + np.exp(L_prev)

    expectation = L_prev + np.log(term_pos) - np.log(term_neg)
    return expectation

def simulate_trial(
    prev_belief_L,
    current_LLR,
    H,
    belief_threshold,
    max_duration_ms,
    dt=0.01,
    noise_std=0.0,
    decision_time_ms=0.0,
    noise_gain=1.0
):
    """
    Simulates a single trial using the continuous Glaze model.

    This function first applies the hazard function (Psi) to the previous belief
    to get the starting belief for the current trial. Then it simulates the
    continuous evolution of belief over time until a threshold is reached or
    time runs out.

    Args:
        prev_belief_L (float): Belief at the end of the previous trial.
        current_LLR (float): Log-likelihood ratio of the evidence for this trial.
        H (float): Hazard rate (subjective or objective).
        belief_threshold (float): Magnitude of belief required to make a decision.
        max_duration_ms (float): Maximum duration of the trial in milliseconds.
        dt (float): Integration time step in seconds.
        noise_std (float): Standard deviation of the Wiener process noise.
        decision_time_ms (float): Minimum time (delay) before a decision can be registered.
        noise_gain (float): Multiplier for the noise magnitude to force threshold crossings.

    Returns:
        dict: A dictionary containing:
            - 'reaction_time_ms': Time at which decision was made or max duration.
            - 'final_belief': Belief at the end of the simulation.
            - 'decision': 1 (positive), -1 (negative), or 0 (timeout).
            - 'trajectory': Array of belief values over time.
            - 'time_points_ms': Array of time points in ms.
    """
    max_duration_sec = max_duration_ms / 1000.0

    # 1. Calculate Prior (The Starting Line) using the discrete hazard update
    psi = psi_function(prev_belief_L, H)

    # Initialize trajectory storage
    trajectory = [psi]
    time_points = [0.0]

    L_current = psi
    time_current = 0.0
    decision = 0

    # 2. Continuous Evolution (The Race)
    while time_current < max_duration_sec:
        # A. The Leak (Stability Term)
        # dL = -2 * lambda * sinh(L) * dt
        # Here we use H as the lambda rate, consistent with Glaze et al. (2015) Eq 4
        deterministic_change = -2 * H * np.sinh(L_current) * dt

        # B. The Evidence (Drift Term)
        drift = current_LLR * dt

        # C. The Noise (Diffusion Term)
        diffusion = noise_gain * noise_std * np.sqrt(dt) * np.random.randn()

        # Update belief and time
        L_current += deterministic_change + drift + diffusion
        time_current += dt

        trajectory.append(L_current)
        time_points.append(time_current * 1000.0)

        # Check for decision threshold crossing
        if abs(L_current) >= belief_threshold and (time_current * 1000.0) >= decision_time_ms:
            decision = np.sign(L_current)
            break

    return {
        'reaction_time_ms': time_current * 1000.0,
        'final_belief': L_current,
        'decision': decision,
        'trajectory': np.array(trajectory),
        'time_points_ms': np.array(time_points)
    }

def plot_model_comparison(df, params=None):
    """
    Plots comparison between real and predicted decisions/RTs.
    """
    # Create a clean dataframe for plotting
    plot_df = df.copy()
    
    # Map predicted decision (-1, 1, 0) to (0, 1, 2)
    # 1 -> 1 (Right/Positive)
    # -1 -> 0 (Left/Negative)
    # 0 -> 2 (Timeout)
    
    y_true = plot_df['choice'].values.astype(int)
    y_pred_raw = plot_df['predicted_decision'].values
    
    y_pred_mapped = []
    agreement = []
    
    for t, p in zip(y_true, y_pred_raw):
        if p == 1:
            p_map = 1
        elif p == -1:
            p_map = 0
        else:
            p_map = 2 # Timeout
            
        y_pred_mapped.append(p_map)
        
        if p_map == 2:
            agreement.append('Timeout')
        elif t == p_map:
            agreement.append('Match')
        else:
            agreement.append('Mismatch')
            
    plot_df['agreement'] = agreement
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if params:
        param_str = " | ".join([f"{k}: {v}" for k, v in params.items()])
        fig.suptitle(f"Model Comparison Parameters\n{param_str}", fontsize=10)
    
    # 1. Confusion Matrix
    conf_matrix = np.zeros((2, 3), dtype=int)
    for t, p in zip(y_true, y_pred_mapped):
        conf_matrix[t, p] += 1
        
    ax1.imshow(conf_matrix, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(3):
            ax1.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black', fontsize=14)
            
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(['Pred 0', 'Pred 1', 'Timeout'])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['True 0', 'True 1'])
    ax1.set_title('Confusion Matrix: Real vs Predicted Choice')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. RT Comparison
    colors = {'Match': 'green', 'Mismatch': 'red', 'Timeout': 'gray'}
    for cat, color in colors.items():
        subset = plot_df[plot_df['agreement'] == cat]
        if not subset.empty:
            ax2.scatter(subset['reaction_time_ms'], subset['predicted_rt_ms'], 
                        c=color, label=cat, alpha=0.6, edgecolors='w', s=60)
            
    max_rt = max(plot_df['reaction_time_ms'].max(), plot_df['predicted_rt_ms'].max())
    ax2.plot([0, max_rt], [0, max_rt], 'k--', alpha=0.5, label='Perfect Fit')
    
    ax2.set_xlabel('Actual Reaction Time (ms)')
    ax2.set_ylabel('Predicted Reaction Time (ms)')
    ax2.set_title('Reaction Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) if params else plt.tight_layout()
    plt.show()

def run_simulation_and_plot(csv_path, block_id=None):
    """
    Runs the simulation and plots the results for a specific block.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    if block_id is not None:
        df = df[df['block_id'] == block_id].copy()

    # Ensure numeric types
    df['reaction_time_ms'] = pd.to_numeric(df['reaction_time_ms'], errors='coerce')
    df['belief_L'] = pd.to_numeric(df['belief_L'], errors='coerce')
    df['LLR'] = pd.to_numeric(df['LLR'], errors='coerce')
    df['subjective_h_snapshot'] = pd.to_numeric(df['subjective_h_snapshot'], errors='coerce')
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce').fillna(0).astype(int)

    # Initialize storage for predictions
    predicted_rts = []
    predicted_decisions = []
    predicted_beliefs = []
    
    # Track belief across trials within blocks
    # We assume the CSV is sorted by block and trial_index
    current_block = None
    prev_actual_belief = 0.0
    prev_actual_cumulative_rt = 0.0
    
    # Parameters for simulation (using typical values or snapshots from CSV)
    # In a real scenario, these might be fitted or taken from 'subjective_h_snapshot'
    threshold = 8.0  # Example threshold
    noise_std = 0.9  # Standard deviation for the Wiener process
    max_duration = 6000  # 5 seconds max
    decision_time_ms = 550.0 # Minimum decision time delay
    noise_gain = 15.5 # Tweak this to force decisions (e.g., set to 2.0 or 3.0)

    # Calculate cumulative RTs for the actual data first to anchor the start times
    # If filtering by block, we can just cumsum the filtered frame if it's a single block
    # or groupby if multiple.
    df['cumulative_rt'] = df.groupby('block_id')['reaction_time_ms'].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    for i, row in df.iterrows():
        # Reset belief if we enter a new block
        if row['block_id'] != current_block:
            current_block = row['block_id']
            prev_actual_belief = 0.0
            prev_actual_cumulative_rt = 0.0
        
        # Run simulation for the current trial
        # Using LLR and subjective_h_snapshot from the CSV
        sim_result = simulate_trial(
            prev_belief_L=prev_actual_belief,
            current_LLR=row['LLR'],
            H=row['subjective_h_snapshot'],
            belief_threshold=threshold,
            max_duration_ms=max_duration,
            noise_std=noise_std,
            decision_time_ms=decision_time_ms,
            noise_gain=noise_gain
        )
        
        # Plot the continuous trajectory
        # The trial starts at the end of the previous trial (prev_actual_cumulative_rt)
        trajectory_times = sim_result['time_points_ms'] + prev_actual_cumulative_rt
        relative_times = sim_result['time_points_ms']
        trajectory_values = sim_result['trajectory']
        
        # Split trajectory into delay period and decision period
        split_idx = np.searchsorted(relative_times, decision_time_ms)
        
        times_pre = trajectory_times[:split_idx]
        values_pre = trajectory_values[:split_idx]
        times_post = trajectory_times[max(0, split_idx-1):]
        values_post = trajectory_values[max(0, split_idx-1):]
        
        if len(times_pre) > 1:
            label_pre = f'Non Decision Period ({decision_time_ms}ms)' if i == df.index[0] else None
            ax1.plot(times_pre, values_pre, color='red', alpha=0.6, label=label_pre)
            
        if len(times_post) > 1:
            label_post = 'Decision Period' if i == df.index[0] else None
            ax1.plot(times_post, values_post, color='green', alpha=0.3, label=label_post)

        predicted_rts.append(sim_result['reaction_time_ms'])
        predicted_decisions.append(sim_result['decision'])
        predicted_beliefs.append(sim_result['final_belief'])
        
        # Update last_belief for the next trial
        # Use the ACTUAL belief and cumulative RT from the data to reset for the next trial (one-step-ahead)
        prev_actual_belief = row['belief_L']
        prev_actual_cumulative_rt = row['cumulative_rt']
        
    # Add predictions back to dataframe for comparison
    df['predicted_rt_ms'] = predicted_rts
    df['predicted_decision'] = predicted_decisions
    df['predicted_belief'] = predicted_beliefs
    
    print(f"Processed {len(df)} trials from {csv_path}")

    # Visualization
    # Calculate predicted cumulative RTs
    # Predicted cumulative RT for trial t is (Cumulative RT at t-1) + Predicted RT at t
    df['prev_cumulative_rt'] = df.groupby('block_id')['cumulative_rt'].shift(1).fillna(0)
    df['predicted_cumulative_rt'] = df['prev_cumulative_rt'] + df['predicted_rt_ms']

    
    # Plot Actual Discrete Beliefs
    ax1.scatter(df['cumulative_rt'], df['belief_L'], label='Actual Discrete Belief', color='blue', alpha=0.6)
    
    # Separate predictions into decisions (threshold crossed) and timeouts
    decision_mask = df['predicted_decision'] != 0
    timeout_mask = df['predicted_decision'] == 0

    # Plot Threshold Crossings
    if decision_mask.any():
        ax1.scatter(
            df.loc[decision_mask, 'predicted_cumulative_rt'], 
            df.loc[decision_mask, 'predicted_belief'], 
            label='Predicted Decision (Belief Threshold)', 
            color='red', 
            marker='x'
        )

    # Plot Timeouts
    if timeout_mask.any():
        ax1.scatter(
            df.loc[timeout_mask, 'predicted_cumulative_rt'], 
            df.loc[timeout_mask, 'predicted_belief'], 
            label=f'Predicted Timeout ({max_duration}ms)', 
            color='orange', 
            marker='s',
            s=30
        )
    
    ax1.set_xlabel('Cumulative Reaction Time (ms)')
    ax1.set_ylabel('Belief (L)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    title = 'Discrete Belief vs. Cumulative Response Time'
    if block_id is not None:
        title += f' (Block {block_id})'
    ax1.set_title(title)
    
    # Add threshold lines
    ax1.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax1.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
    
    # Secondary Axis for Subjective H
    ax2 = ax1.twinx()
    ax2.plot(df['cumulative_rt'], df['subjective_h_snapshot'], color='magenta', linestyle=':', marker='.', alpha=0.5, label='Subjective H')
    ax2.set_ylabel('Subjective Hazard Rate', color='magenta')
    ax2.tick_params(axis='y', labelcolor='magenta')
    ax2.set_ylim(0, 1)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax1.grid(True, alpha=0.3)
    plt.show()

    # Plot comparison
    sim_params = {
        'Block': block_id,
        'Noise Gain': noise_gain,
        'Max Duration': max_duration,
        'Decision Time': decision_time_ms,
        'Threshold': threshold
    }
    plot_model_comparison(df, params=sim_params)

if __name__ == '__main__':
    # Load the data from the specified CSV path
    csv_path = './triangle-data/evan-standard.csv'
    block_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run_simulation_and_plot(csv_path, block_id=block_id)