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
        # Clamp L for sinh calculation to prevent numerical explosion (instability)
        # sinh(10) is ~11000, which is plenty of restoring force without overflowing with dt=0.01
        L_safe = np.clip(L_current, -10, 10)
        deterministic_change = -2 * H * np.sinh(L_safe) * dt

        # B. The Evidence (Drift Term)
        drift = current_LLR * dt

        # C. The Noise (Diffusion Term)
        diffusion = noise_gain * noise_std * np.sqrt(dt) * np.random.randn()

        # Update belief and time
        L_current += deterministic_change + drift + diffusion
        
        # Clamp belief to prevent plot scaling issues if instability occurs
        L_current = np.clip(L_current, -100, 100)
        
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

    # Calculate Metrics
    # Correct: (True=1 & Pred=1) or (True=0 & Pred=-1)
    correct_mask = ((y_true == 1) & (y_pred_raw == 1)) | ((y_true == 0) & (y_pred_raw == -1))
    accuracy = np.mean(correct_mask)
    
    # Precision/Recall for Class 1 (Right)
    tp = np.sum((y_true == 1) & (y_pred_raw == 1))
    fp = np.sum((y_true == 0) & (y_pred_raw == 1))
    fn = np.sum((y_true == 1) & (y_pred_raw != 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    metrics_str = f"Accuracy: {accuracy:.2%} | Precision: {precision:.2f} | Recall: {recall:.2f} | F1: {f1:.2f}"
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    
    if params:
        param_str = " | ".join([f"{k}: {v}" for k, v in params.items()])
        fig.suptitle(f"Model Comparison Parameters\n{param_str}\n{metrics_str}", fontsize=12)
    else:
        fig.suptitle(f"Model Performance Metrics\n{metrics_str}", fontsize=12)
    
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
    # Color by Block ID
    unique_blocks = sorted(plot_df['block_id'].unique())
    cmap = plt.get_cmap('tab10')
    
    for i, block in enumerate(unique_blocks):
        subset = plot_df[plot_df['block_id'] == block]
        color = cmap(i % 10)
        ax2.scatter(subset['reaction_time_ms'], subset['predicted_rt_ms'], 
                    color=color, label=f'Block {block}', alpha=0.6, edgecolors='w', s=60)
            
    max_rt = max(plot_df['reaction_time_ms'].max(), plot_df['predicted_rt_ms'].max())
    ax2.plot([0, max_rt], [0, max_rt], 'k--', alpha=0.5, label='Perfect Fit')
    
    ax2.set_xlabel('Actual Reaction Time (ms)')
    ax2.set_ylabel('Predicted Reaction Time (ms)')
    ax2.set_title('Reaction Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Belief Comparison
    colors = {'Match': 'green', 'Mismatch': 'red', 'Timeout': 'gray'}
    for cat, color in colors.items():
        subset = plot_df[plot_df['agreement'] == cat]
        if not subset.empty:
            ax3.scatter(subset['belief_L'], subset['predicted_belief'], 
                        c=color, label=cat, alpha=0.6, edgecolors='w', s=60)
    
    # Identity line
    min_b = min(plot_df['belief_L'].min(), plot_df['predicted_belief'].min())
    max_b = max(plot_df['belief_L'].max(), plot_df['predicted_belief'].max())
    ax3.plot([min_b, max_b], [min_b, max_b], 'k--', alpha=0.5, label='Identity')
    
    ax3.set_xlabel('Actual Belief (L)')
    ax3.set_ylabel('Predicted Belief (L)')
    ax3.set_title('Belief Magnitude Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Psychometric Curve (Choice vs LLR)
    # Bin LLRs
    if not plot_df.empty:
        llr_min, llr_max = plot_df['LLR'].min(), plot_df['LLR'].max()
        # Create 8 bins
        bins = np.linspace(llr_min, llr_max, 9)
        plot_df['llr_bin'] = pd.cut(plot_df['LLR'], bins)
        
        # Actual Probabilities (Choice 1)
        actual_probs = plot_df.groupby('llr_bin', observed=False)['choice'].mean()
        bin_centers_act = [(b.left + b.right)/2 for b in actual_probs.index]
        ax4.plot(bin_centers_act, actual_probs, 'o-', label='Actual Data', color='blue')
        
        # Predicted Probabilities (Decision 1) - Exclude Timeouts for curve
        pred_valid = plot_df[plot_df['predicted_decision'] != 0].copy()
        if not pred_valid.empty:
            pred_valid['pred_choice_mapped'] = (pred_valid['predicted_decision'] == 1).astype(int)
            pred_probs = pred_valid.groupby('llr_bin', observed=False)['pred_choice_mapped'].mean()
            bin_centers_pred = [(b.left + b.right)/2 for b in pred_probs.index]
            ax4.plot(bin_centers_pred, pred_probs, 's--', label='Model Prediction', color='red')
        
        ax4.set_xlabel('Evidence (LLR)')
        ax4.set_ylabel('P(Choice = Right)')
        ax4.set_title('Psychometric Curve (Choice Probability)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) if params else plt.tight_layout()
    plt.show()

def run_simulation_and_plot(csv_path, block_id=None):
    """
    Runs the simulation and plots the results for a specific block.
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Ensure numeric types
    df['reaction_time_ms'] = pd.to_numeric(df['reaction_time_ms'], errors='coerce')
    df['belief_L'] = pd.to_numeric(df['belief_L'], errors='coerce')
    df['LLR'] = pd.to_numeric(df['LLR'], errors='coerce')
    df['subjective_h_snapshot'] = pd.to_numeric(df['subjective_h_snapshot'], errors='coerce')
    df['choice'] = pd.to_numeric(df['choice'], errors='coerce').fillna(0).astype(int)

    # Parameters for simulation (using typical values or snapshots from CSV)
    # In a real scenario, these might be fitted or taken from 'subjective_h_snapshot'
    # threshold is now calculated per block
    noise_std = 0.9  # Standard deviation for the Wiener process
    max_duration = 6000  # 5 seconds max
    decision_time_ms = 550.0 # Minimum decision time delay
    noise_gain = 15.5 # Tweak this to force decisions (e.g., set to 2.0 or 3.0)

    # Calculate cumulative RTs for the actual data first to anchor the start times
    # If filtering by block, we can just cumsum the filtered frame if it's a single block
    # or groupby if multiple.
    df['cumulative_rt'] = df.groupby('block_id')['reaction_time_ms'].cumsum()

    # Determine which blocks to process
    if block_id is not None:
        blocks_to_process = [block_id]
    else:
        blocks_to_process = sorted(df['block_id'].unique())

    processed_blocks = []

    for b_id in blocks_to_process:
        # Filter for the current block
        block_df = df[df['block_id'] == b_id].copy()
        
        # Calculate threshold dynamically for this block
        threshold = block_df['belief_L'].abs().mean()
        
        # Initialize storage for predictions
        predicted_rts = []
        predicted_decisions = []
        predicted_beliefs = []
        
        prev_actual_belief = 0.0
        prev_actual_cumulative_rt = 0.0
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        for i, row in block_df.iterrows():
            # Run simulation for the current trial
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
                label_pre = f'Non Decision Period ({decision_time_ms}ms)' if i == block_df.index[0] else None
                ax1.plot(times_pre, values_pre, color='red', alpha=0.6, label=label_pre)
                
            if len(times_post) > 1:
                label_post = 'Decision Period' if i == block_df.index[0] else None
                ax1.plot(times_post, values_post, color='green', alpha=0.3, label=label_post)

            predicted_rts.append(sim_result['reaction_time_ms'])
            predicted_decisions.append(sim_result['decision'])
            predicted_beliefs.append(sim_result['final_belief'])
            
            # Update last_belief for the next trial
            prev_actual_belief = row['belief_L']
            prev_actual_cumulative_rt = row['cumulative_rt']
            
        # Add predictions back to dataframe
        block_df['predicted_rt_ms'] = predicted_rts
        block_df['predicted_decision'] = predicted_decisions
        block_df['predicted_belief'] = predicted_beliefs
        
        # Calculate predicted cumulative RTs for plotting
        block_df['prev_cumulative_rt'] = block_df['cumulative_rt'].shift(1).fillna(0)
        block_df['predicted_cumulative_rt'] = block_df['prev_cumulative_rt'] + block_df['predicted_rt_ms']
        
        # --- Plotting for this block ---
        ax1.scatter(block_df['cumulative_rt'], block_df['belief_L'], label='Actual Discrete Belief', color='blue', alpha=0.6)
        
        decision_mask = block_df['predicted_decision'] != 0
        timeout_mask = block_df['predicted_decision'] == 0

        if decision_mask.any():
            ax1.scatter(block_df.loc[decision_mask, 'predicted_cumulative_rt'], block_df.loc[decision_mask, 'predicted_belief'], 
                        label='Predicted Decision', color='red', marker='x')
        if timeout_mask.any():
            ax1.scatter(block_df.loc[timeout_mask, 'predicted_cumulative_rt'], block_df.loc[timeout_mask, 'predicted_belief'], 
                        label='Predicted Timeout', color='orange', marker='s', s=30)
        
        ax1.set_xlabel('Cumulative Reaction Time (ms)')
        ax1.set_ylabel('Belief (L)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_title(f'Discrete Belief vs. Cumulative Response Time (Block {b_id})')
        ax1.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax1.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
        
        ax2 = ax1.twinx()
        ax2.plot(block_df['cumulative_rt'], block_df['subjective_h_snapshot'], color='magenta', linestyle=':', marker='.', alpha=0.5, label='Subjective H')
        ax2.set_ylabel('Subjective Hazard Rate', color='magenta')
        ax2.tick_params(axis='y', labelcolor='magenta')
        ax2.set_ylim(0, 1)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        plt.show()
        
        processed_blocks.append(block_df)
        print(f"Processed Block {b_id}: {len(block_df)} trials")

    # Concatenate all processed blocks for comparison
    full_df = pd.concat(processed_blocks)

    sim_params = {
        'Block': 'All' if block_id is None else block_id,
        'Noise Gain': noise_gain,
        'Max Duration': max_duration,
        'Decision Time': decision_time_ms,
        'Threshold': f"{threshold:.2f}" if len(blocks_to_process) == 1 else "Dynamic (Mean |L|)"
    }
    plot_model_comparison(full_df, params=sim_params)

if __name__ == '__main__':
    # Load the data from the specified CSV path
    #csv_path = './triangle-data/evan-standard.csv'
    csv_path = 'data/participants.csv'
    block_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_simulation_and_plot(csv_path, block_id=block_id)