import os
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
    dt=0.01
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

        # Update belief and time
        L_current += deterministic_change + drift
        time_current += dt

        trajectory.append(L_current)
        time_points.append(time_current * 1000.0)

        # Check for decision threshold crossing
        if abs(L_current) >= belief_threshold:
            decision = np.sign(L_current)
            break

    return {
        'reaction_time_ms': time_current * 1000.0,
        'final_belief': L_current,
        'decision': decision,
        'trajectory': np.array(trajectory),
        'time_points_ms': np.array(time_points)
    }

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
    threshold = 5.0  # Example threshold
    max_duration = 10000  # 5 seconds max

    # Calculate cumulative RTs for the actual data first to anchor the start times
    # If filtering by block, we can just cumsum the filtered frame if it's a single block
    # or groupby if multiple.
    df['cumulative_rt'] = df.groupby('block_id')['reaction_time_ms'].cumsum()

    plt.figure(figsize=(12, 6))
    
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
            max_duration_ms=max_duration
        )
        
        # Plot the continuous trajectory
        # The trial starts at the end of the previous trial (prev_actual_cumulative_rt)
        trajectory_times = sim_result['time_points_ms'] + prev_actual_cumulative_rt
        
        # Only add label to the first segment to avoid legend duplication
        label = 'Continuous Trajectory' if i == df.index[0] else None
        plt.plot(trajectory_times, sim_result['trajectory'], color='green', alpha=0.3, label=label)

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
    plt.scatter(df['cumulative_rt'], df['belief_L'], label='Actual Discrete Belief', color='blue', alpha=0.6)
    
    # Separate predictions into decisions (threshold crossed) and timeouts
    decision_mask = df['predicted_decision'] != 0
    timeout_mask = df['predicted_decision'] == 0

    # Plot Threshold Crossings
    if decision_mask.any():
        plt.scatter(
            df.loc[decision_mask, 'predicted_cumulative_rt'], 
            df.loc[decision_mask, 'predicted_belief'], 
            label='Predicted Decision (Threshold)', 
            color='red', 
            marker='x'
        )

    # Plot Timeouts
    if timeout_mask.any():
        plt.scatter(
            df.loc[timeout_mask, 'predicted_cumulative_rt'], 
            df.loc[timeout_mask, 'predicted_belief'], 
            label='Predicted Timeout', 
            color='orange', 
            marker='s',
            s=30
        )
    
    plt.xlabel('Cumulative Reaction Time (ms)')
    plt.ylabel('Belief (L)')
    
    title = 'Discrete Belief vs. Cumulative Response Time'
    if block_id is not None:
        title += f' (Block {block_id})'
    plt.title(title)
    
    # Add threshold lines
    plt.axhline(y=threshold, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    plt.axhline(y=-threshold, color='gray', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == '__main__':
    # Load the data from the specified CSV path
    csv_path = './triangle-data/evan-standard.csv'
    run_simulation_and_plot(csv_path, block_id=1)
