"""
Wheel Rolling Experiment Graphing Module

This module provides functions to generate publication-quality graphs
for wheel rolling experiments, including:
  - Sensor voltage readings over time
  - Position, velocity, and acceleration kinematics
  - Summary statistics output to terminal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Constants
THRESHOLD = 3.0  # Default voltage threshold (V)
SENSOR_SPACING = 0.3  # Default spacing between sensors (m)
FLAT_REGION_SENSORS = list(range(2, 7)) + list(range(9, 15))  # Sensors on flat track sections

def load_sensor_data():
    """
    Load sensor configuration from sensor_data.csv.
    
    Returns:
        Dictionary mapping sensor number to {distance_m, position_m, activation_threshold, is_ramp}
        where position_m is the distance along track and distance_m is horizontal offset.
    """
    try:
        sensor_data_path = Path('data/sensor_data.csv')
        if not sensor_data_path.exists():
            print("[WARNING] sensor_data.csv not found, using default spacing")
            return None
        
        df = pd.read_csv(sensor_data_path, index_col=False)
        sensor_info = {}
        
        for _, row in df.iterrows():
            sensor_num = int(row['sensor'])
            distance_cm = float(row['distance(cm)'])
            height_cm = float(row['height(cm)'])
            activation_volt = float(row['activationvolt'])
            is_ramp = bool(row['ramp'])
            
            sensor_info[sensor_num] = {
                'distance_m': distance_cm / 100.0,
                'position_m': distance_cm / 100.0,  # Use distance along track as position
                'height_m': height_cm / 100.0,  # Keep height separate for reference
                'activation_threshold': activation_volt,
                'is_ramp': is_ramp
            }
        
        print(f"[INFO] Loaded position data for {len(sensor_info)} sensors")
        return sensor_info
    except Exception as e:
        print(f"[WARNING] Could not load sensor_data.csv: {e}")
        return None

SENSOR_DATA = load_sensor_data()

def find_critical_activation_times(time_data, voltage_data, threshold=THRESHOLD):
    """
    Find activation times for sensor peaks using threshold crossing detection.
    
    Critical time is defined as the midpoint between threshold entry and exit.
    Returns all distinct crossing events and their boundary times.
    
    Args:
        time_data: Array of time values
        voltage_data: Array of voltage values
        threshold: Voltage threshold for detection
    
    Returns:
        Tuple of (critical_times_array, first_entry_time, last_exit_time)
    """
    above_threshold = voltage_data > threshold
    
    if not above_threshold.any():
        return np.array([]), None, None
    
    critical_times = []
    entry_times = []
    exit_times = []
    in_crossing = False
    crossing_start_idx = None
    
    for i in range(len(above_threshold)):
        if above_threshold[i] and not in_crossing:
            in_crossing = True
            crossing_start_idx = i
        elif not above_threshold[i] and in_crossing:
            crossing_end_idx = i - 1
            entry_time = time_data[crossing_start_idx]
            exit_time = time_data[crossing_end_idx]
            critical_time = (entry_time + exit_time) / 2.0
            critical_times.append(critical_time)
            entry_times.append(entry_time)
            exit_times.append(exit_time)
            in_crossing = False
    
    if in_crossing:
        entry_time = time_data[crossing_start_idx]
        exit_time = time_data[-1]
        critical_time = (entry_time + exit_time) / 2.0
        critical_times.append(critical_time)
        entry_times.append(entry_time)
        exit_times.append(exit_time)
    
    first_entry = min(entry_times) if entry_times else None
    last_exit = max(exit_times) if exit_times else None
    
    return np.array(critical_times), first_entry, last_exit


def find_crossing_time(time_data, voltage_data, threshold=THRESHOLD, use_first=False):
    """
    Find the time when the wheel passes through a sensor.
    
    Args:
        time_data: Array of time values
        voltage_data: Array of voltage values for the sensor
        threshold: Voltage threshold for detection
        use_first: If True, use first crossing time; if False, use time of max voltage
    
    Returns:
        Crossing time in seconds, or None if sensor not triggered
    """
    above_threshold = voltage_data > threshold
    
    if not above_threshold.any():
        return None
    
    if use_first:
        first_idx = np.where(above_threshold)[0][0]
        crossing_time = time_data[first_idx]
    else:
        max_idx = voltage_data.argmax()
        crossing_time = time_data[max_idx]
    
    return crossing_time


def plot_sensor_voltages(data, time, filename=None, save=None):
    """
    Generate sensor voltage vs time plot for all sensors.
    
    Args:
        data: DataFrame with sensor columns
        time: Time array
        filename: Base filename (without extension) for the output file
        save: Directory path to save the figure. If None, plot is not saved.
              Can be a directory path (string or Path) where file will be saved.
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    
    colors = plt.cm.tab20(np.linspace(0, 1, 16))
    
    for sensor_num in range(1, 17):
        sensor_col = f'Sens{sensor_num}'
        if sensor_col in data.columns:
            voltage = data[sensor_col].values
            ax.plot(time, voltage, label=f'Sensor {sensor_num}', 
                   color=colors[sensor_num-1], alpha=0.7, linewidth=1.5)
    
    # Note: Custom thresholds from sensor_data.csv are used for activation detection
    # but are not shown on the plot since each sensor has a different threshold
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Voltage (V)', fontsize=14, fontweight='bold')
    ax.set_title('Sensor Readings - Voltage vs Time', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    if save is not None:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = "sensor_data"
        output_path = save_dir / f"{filename}_voltage_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Sensor voltage plot saved: {output_path}")
    
    return fig


def plot_full_experiment(all_sensor_data, time_full, filename=None, save=None, experiment_start_time=None, experiment_end_time=None):
    """
    Generate position, speed, and acceleration plots for full experiment timeline.
    
    Speed calculation uses centered difference with special handling for oscillations
    (consecutive readings from the same sensor). Acceleration is derived from the
    quadratic speed fit curve.
    
    Args:
        all_sensor_data: Dictionary mapping sensor number to activation times
        time_full: Complete time array from raw data
        filename: Base filename for output (without extension)
        save: Directory path to save figure, or None to skip saving
        experiment_start_time: First threshold crossing time across all sensors
        experiment_end_time: Last threshold crossing time across all sensors
    
    Returns:
        Tuple of (average_acceleration, r_squared, figure_object)
    """
    all_activations = []
    
    for sensor_num in range(1, 17):
        if sensor_num in all_sensor_data and len(all_sensor_data[sensor_num]) > 0:
            times = all_sensor_data[sensor_num]
            if SENSOR_DATA and sensor_num in SENSOR_DATA:
                position = SENSOR_DATA[sensor_num]['position_m']
                is_ramp = SENSOR_DATA[sensor_num]['is_ramp']
            else:
                position = sensor_num * SENSOR_SPACING
                is_ramp = sensor_num not in FLAT_REGION_SENSORS
            
            for t in times:
                all_activations.append((t, sensor_num, position, is_ramp))
    
    all_activations.sort(key=lambda x: x[0])
    
    if len(all_activations) == 0:
        print("[WARNING] No sensor activations found")
        return 0, 0, None
    
    times_seq = np.array([a[0] for a in all_activations])
    sensors_seq = np.array([a[1] for a in all_activations])
    positions_seq = np.array([a[2] for a in all_activations])
    is_ramp_seq = np.array([a[3] for a in all_activations])
    
    speeds = []
    speed_times = []
    speed_sensors = []
    speed_is_ramp = []
    speed_is_oscillation = []
    
    i = 0
    while i < len(times_seq):
        current_sensor = sensors_seq[i]
        current_time = times_seq[i]
        current_pos = positions_seq[i]
        current_is_ramp = is_ramp_seq[i]
        
        if i == 0:
            if len(times_seq) > 1:
                dt = times_seq[1] - times_seq[0]
                dx = positions_seq[1] - positions_seq[0]
                if dt > 0:
                    speed = abs(dx / dt)
                    speeds.append(speed)
                    speed_times.append(current_time)
                    speed_sensors.append(current_sensor)
                    speed_is_ramp.append(current_is_ramp)
                    speed_is_oscillation.append(False)
            i += 1
            
        elif i == len(times_seq) - 1:
            dt = times_seq[i] - times_seq[i-1]
            dx = positions_seq[i] - positions_seq[i-1]
            if dt > 0:
                speed = abs(dx / dt)
                speeds.append(speed)
                speed_times.append(current_time)
                speed_sensors.append(current_sensor)
                speed_is_ramp.append(current_is_ramp)
                speed_is_oscillation.append(False)
            i += 1
            
        elif i + 1 < len(times_seq) and sensors_seq[i + 1] == current_sensor:
            next_time = times_seq[i + 1]
            avg_time = (current_time + next_time) / 2.0
            
            if i > 0:
                dt_pre = avg_time - times_seq[i-1]
                dx_pre = current_pos - positions_seq[i-1]
                speed_pre = abs(dx_pre / dt_pre) if dt_pre > 0 else 0
            else:
                speed_pre = 0
            
            if i + 2 < len(times_seq):
                dt_post = times_seq[i+2] - avg_time
                dx_post = positions_seq[i+2] - current_pos
                speed_post = abs(dx_post / dt_post) if dt_post > 0 else 0
            else:
                speed_post = 0
            
            if speed_pre > 0 and speed_post > 0:
                speed = (speed_pre + speed_post) / 2.0
            elif speed_pre > 0:
                speed = speed_pre
            elif speed_post > 0:
                speed = speed_post
            else:
                speed = 0
            
            if speed > 0:
                speeds.append(speed)
                speed_times.append(avg_time)
                speed_sensors.append(current_sensor)
                speed_is_ramp.append(current_is_ramp)
                speed_is_oscillation.append(True)
            
            i += 2
            
        else:
            dt_pre = current_time - times_seq[i-1]
            dx_pre = current_pos - positions_seq[i-1]
            speed_pre = abs(dx_pre / dt_pre) if dt_pre > 0 else 0
            
            dt_post = times_seq[i+1] - current_time
            dx_post = positions_seq[i+1] - current_pos
            speed_post = abs(dx_post / dt_post) if dt_post > 0 else 0
            
            if speed_pre > 0 and speed_post > 0:
                speed = (speed_pre + speed_post) / 2.0
            elif speed_pre > 0:
                speed = speed_pre
            elif speed_post > 0:
                speed = speed_post
            else:
                speed = 0
            
            if speed > 0:
                speeds.append(speed)
                speed_times.append(current_time)
                speed_sensors.append(current_sensor)
                speed_is_ramp.append(current_is_ramp)
                speed_is_oscillation.append(False)
            
            i += 1
    
    speeds = np.array(speeds)
    speed_times = np.array(speed_times)
    speed_sensors = np.array(speed_sensors)
    speed_is_ramp = np.array(speed_is_ramp)
    speed_is_oscillation = np.array(speed_is_oscillation)
    
    print(f"[INFO] Calculated speeds for {len(speeds)} sensor activations")
    print(f"[INFO] Oscillation events: {np.sum(speed_is_oscillation)}")
    
    flat_mask_speed = ~speed_is_ramp
    speed_poly_coeffs = None
    accel_from_fit = None
    accel_from_fit_times = None
    
    if np.any(flat_mask_speed) and np.sum(flat_mask_speed) > 2:
        speeds_flat_fit = speeds[flat_mask_speed]
        times_flat_fit = speed_times[flat_mask_speed]
        speed_poly_coeffs = np.polyfit(times_flat_fit, speeds_flat_fit, 2)
        
        accel_poly_coeffs = np.polyder(speed_poly_coeffs)
        
        accel_from_fit = []
        accel_from_fit_times = []
        
        for i in range(len(speed_times)):
            accel_at_time = np.polyval(accel_poly_coeffs, speed_times[i])
            accel_from_fit.append(accel_at_time)
            accel_from_fit_times.append(speed_times[i])
        
        accel_from_fit = np.array(accel_from_fit)
        accel_from_fit_times = np.array(accel_from_fit_times)
        
        print(f"[INFO] Calculated accelerations from speed fit for {len(accel_from_fit)} sensor activations")
    
    if speed_poly_coeffs is not None and accel_from_fit is not None:
        avg_accel_flat = np.mean(accel_from_fit)
        
        flat_mask = ~speed_is_ramp
        speeds_flat = speeds[flat_mask]
        speed_times_flat = speed_times[flat_mask]
        
        if len(speeds_flat) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(speed_times_flat, speeds_flat)
            r_squared_flat = r_value ** 2
        else:
            r_squared_flat = 0
    else:
        avg_accel_flat = 0
        r_squared_flat = 0
    
    if experiment_start_time is not None:
        t_min = max(0, experiment_start_time - 1.0)
    else:
        t_min = max(0, times_seq.min() - 1.0)
    
    if experiment_end_time is not None:
        t_max = min(time_full[-1], experiment_end_time + 1.0)
    else:
        t_max = min(time_full[-1], times_seq.max() + 1.0)
    
    t_plot = np.linspace(t_min, t_max, 1000)
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    ax1 = axes[0]
    flat_mask_pos = ~is_ramp_seq
    ramp_mask_pos = is_ramp_seq
    
    if np.any(flat_mask_pos):
        ax1.scatter(times_seq[flat_mask_pos], positions_seq[flat_mask_pos], 
                   s=50, c='blue', marker='o', alpha=0.6, 
                   edgecolors='black', linewidth=0.5, label='Flat regions', zorder=3)
    
    if np.any(ramp_mask_pos):
        ax1.scatter(times_seq[ramp_mask_pos], positions_seq[ramp_mask_pos], 
                   s=50, c='orange', marker='o', alpha=0.6, 
                   edgecolors='black', linewidth=0.5, label='Ramps', zorder=3)
    
    ax1.set_ylabel('Position (m)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Full Experiment Kinematics - Complete Timeline ({t_min:.1f}s to {t_max:.1f}s)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    ax1.set_xlim(t_min, t_max)
    
    ax2 = axes[1]
    
    if len(speeds) > 0:
        normal_mask = ~speed_is_oscillation
        oscillation_mask = speed_is_oscillation
        
        if np.any(normal_mask):
            ax2.scatter(speed_times[normal_mask], speeds[normal_mask], 
                       s=50, c='green', marker='s', alpha=0.6, 
                       edgecolors='black', linewidth=0.5, 
                       label='Normal speed', zorder=3)
        
        if np.any(oscillation_mask):
            ax2.scatter(speed_times[oscillation_mask], speeds[oscillation_mask], 
                       s=50, c='red', marker='D', alpha=0.6, 
                       edgecolors='black', linewidth=0.5, 
                       label='Oscillation speed', zorder=4)
        
        if speed_poly_coeffs is not None:
            speed_fit = np.polyval(speed_poly_coeffs, t_plot)
            ax2.plot(t_plot, speed_fit, 'b--', linewidth=2, 
                    label=f'Best fit (flat): quadratic', zorder=2)
    
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Speed (m/s)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    ax2.set_xlim(t_min, t_max)
    
    ax3 = axes[2]
    
    if accel_from_fit is not None and len(accel_from_fit) > 0:
        avg_accel_simple = np.mean(accel_from_fit)
        
        ax3.scatter(accel_from_fit_times, accel_from_fit, 
                   s=50, c='orange', marker='o', alpha=0.6, 
                   edgecolors='black', linewidth=0.5, 
                   label=f'Acceleration from speed fit (Avg: {avg_accel_simple:.6f} m/s²)', zorder=3)
        
        if len(accel_from_fit) > 1:
            accel_fit_coeffs = np.polyfit(accel_from_fit_times, accel_from_fit, 1)
            accel_fit_line = np.polyval(accel_fit_coeffs, t_plot)
            
            ax3.plot(t_plot, accel_fit_line, 'r-', linewidth=2, 
                    label=f'Best fit', zorder=2)
    
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=12)
    ax3.set_xlim(t_min, t_max)
    
    plt.tight_layout()
    
    if save is not None:
        save_dir = Path(save)
        save_dir.mkdir(parents=True, exist_ok=True)
        if filename is None:
            filename = "full_experiment"
        output_path = save_dir / f"{filename}_posvelacc_time_full.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Full experiment kinematics plot saved: {output_path}")
    
    return avg_accel_flat, r_squared_flat, fig


def calculate_and_print_metrics(times_array, positions_array, velocities, 
                                acceleration, r_squared, filename,
                                experiment_start_time=None, experiment_end_time=None,
                                full_experiment_acceleration=None):
    """
    Calculate and print experiment metrics to terminal.
    
    Args:
        times_array: Sensor crossing times
        positions_array: Sensor positions
        velocities: Calculated velocities
        acceleration: Average acceleration
        r_squared: Goodness of fit
        filename: Data file name
        experiment_start_time: Start time of full experiment
        experiment_end_time: End time of full experiment
        full_experiment_acceleration: Acceleration from full experiment
    """
    duration = times_array.max() - times_array.min()
    distance = positions_array.max() - positions_array.min()
    initial_velocity = velocities[0] if len(velocities) > 0 else None
    final_velocity = velocities[-1] if len(velocities) > 0 else None
    
    full_duration = None
    if experiment_start_time is not None and experiment_end_time is not None:
        full_duration = experiment_end_time - experiment_start_time
    
    print("\n" + "="*70)
    print(f"Analysis Results: {filename}")
    print("="*70)
    print(f"Experiment Duration:        {duration:.4f} seconds")
    print(f"Distance Traveled:          {distance:.4f} meters")
    print(f"Average Acceleration:       {acceleration:.4f} m/s²")
    print(f"Initial Velocity:           {initial_velocity:.4f} m/s" if initial_velocity else "Initial Velocity:           N/A")
    print(f"Final Velocity:             {final_velocity:.4f} m/s" if final_velocity else "Final Velocity:             N/A")
    print(f"R² (velocity fit):          {r_squared:.4f}")
    print(f"Number of Data Points:      {len(times_array)}")
    print("="*70)
    
    return {
        'duration_s': full_duration if full_duration is not None else duration,
        'distance_m': distance,
        'acceleration_ms2': full_experiment_acceleration if full_experiment_acceleration is not None else acceleration,
        'initial_velocity_ms': initial_velocity,
        'final_velocity_ms': final_velocity,
        'r_squared': r_squared,
        'num_sensors': len(times_array)
    }


def graph_on_file(csv_file_path, output_dir=None, save=True):
    """
    Process CSV data and generate kinematics plots.
    
    Args:
        csv_file_path: Path to CSV file
        output_dir: Output directory (uses input directory if None)
        save: True to save plots, False to display only, or path string
    
    Returns:
        Dictionary with calculated metrics
    """
    csv_path = Path(csv_file_path)
    
    if not csv_path.exists():
        print(f"[ERROR] File not found: {csv_path}")
        return None
    
    print("\n" + "="*70)
    print(f"Processing: {csv_path.name}")
    print("="*70)
    
    if save is True:
        save_dir = Path(output_dir) if output_dir else csv_path.parent
    elif save is False:
        save_dir = None
    else:
        save_dir = Path(save)
    
    base_name = csv_path.stem
    
    try:
        with open(csv_path, 'r') as f:
            first_line = f.readline()
        
        if 'Channels' in first_line or 'LabVIEW' in first_line:
            data = pd.read_csv(csv_path, skiprows=8, index_col=False)
        else:
            data = pd.read_csv(csv_path, index_col=False)
        
        if 'X_Value' not in data.columns:
            print(f"[ERROR] No X_Value column found in {csv_path.name}")
            return None
        
        time = data['X_Value'].values
        
        print(f"[INFO] Loaded {len(time)} data points")
        print(f"[INFO] Time range: {time.min():.3f}s to {time.max():.3f}s")
        
        print("\n[PROCESSING] Generating sensor voltage plot...")
        fig_sensors = plot_sensor_voltages(data, time, filename=base_name, save=save_dir)
        
        print("\n[PROCESSING] Detecting sensor crossings...")
        sensor_times_first = {}
        sensor_times_all = {}
        sensor_positions = {}
        sensors_triggered = []
        all_first_entries = []
        all_last_exits = []
        
        for sensor_num in range(1, 17):
            sensor_col = f'Sens{sensor_num}'
            if sensor_col not in data.columns:
                continue
            
            voltage = data[sensor_col].values
            
            if SENSOR_DATA and sensor_num in SENSOR_DATA:
                activation_threshold = SENSOR_DATA[sensor_num].get('activation_threshold', THRESHOLD)
            else:
                activation_threshold = THRESHOLD
            
            crossing_time_first = find_crossing_time(time, voltage, activation_threshold, use_first=True)
            all_critical_times, first_entry, last_exit = find_critical_activation_times(time, voltage, activation_threshold)
            
            if crossing_time_first is not None:
                sensor_times_first[sensor_num] = crossing_time_first
                sensor_times_all[sensor_num] = all_critical_times
                if SENSOR_DATA and sensor_num in SENSOR_DATA:
                    sensor_positions[sensor_num] = SENSOR_DATA[sensor_num]['position_m']
                else:
                    sensor_positions[sensor_num] = sensor_num * SENSOR_SPACING
                sensors_triggered.append(sensor_num)
                
                if first_entry is not None:
                    all_first_entries.append(first_entry)
                if last_exit is not None:
                    all_last_exits.append(last_exit)
        
        print(f"[INFO] Sensors triggered: {sensors_triggered}")
        
        experiment_start_time = min(all_first_entries) if all_first_entries else None
        experiment_end_time = max(all_last_exits) if all_last_exits else None
        
        if experiment_start_time is not None and experiment_end_time is not None:
            print(f"[INFO] Experiment range: {experiment_start_time:.2f}s to {experiment_end_time:.2f}s")
        
        print("\n[PROCESSING] Generating full experiment kinematics plot...")
        avg_acceleration_full, r_squared_full, fig_kinematics_full = plot_full_experiment(
            sensor_times_all, time, filename=base_name, save=save_dir,
            experiment_start_time=experiment_start_time, experiment_end_time=experiment_end_time
        )
        
        metrics = {
            'duration_s': experiment_end_time - experiment_start_time,
            'acceleration_ms2': avg_acceleration_full,
            'r_squared': r_squared_full
        }
        
        print(f"\n[INFO] Experiment Duration: {metrics['duration_s']:.4f} seconds")
        print(f"[INFO] Average Acceleration: {metrics['acceleration_ms2']:.4f} m/s²")
        print(f"[INFO] R²: {metrics['r_squared']:.4f}")
        
        print("\n[SUCCESS] Processing complete!")
        print("="*70)
        
        if save_dir is not None:
            plt.close(fig_sensors)
            plt.close(fig_kinematics_full)
        
        return metrics
        
    except Exception as e:
        print(f"\n[ERROR] Failed to process file: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_aggregate_runs(csv_files, output_path, experiment_name):
    """
    Generate aggregate position, velocity, and acceleration plots for multiple runs.
    All runs are plotted on the same axes with t=0 as the experiment start time.
    
    Args:
        csv_files: List of paths to CSV files for the same experiment group
        output_path: Path where the output PNG will be saved
        experiment_name: Name of the experiment (e.g., "Directional 5mm")
    
    Returns:
        True if successful, False otherwise
    """
    if not csv_files:
        print("[ERROR] No CSV files provided")
        return False
    
    print(f"\n[INFO] Processing {len(csv_files)} runs for {experiment_name}")
    
    # Data storage for all runs
    all_runs_data = []
    
    # Process each CSV file
    for csv_file in csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"[WARNING] File not found: {csv_file}")
            continue
        
        try:
            # Load CSV data
            with open(csv_path, 'r') as f:
                first_line = f.readline()
            
            if 'Channels' in first_line or 'LabVIEW' in first_line:
                data = pd.read_csv(csv_path, skiprows=8, index_col=False)
            else:
                data = pd.read_csv(csv_path, index_col=False)
            
            if 'X_Value' not in data.columns:
                print(f"[WARNING] No X_Value column in {csv_file}")
                continue
            
            time = data['X_Value'].values
            
            # Detect sensor crossings
            sensor_times_all = {}
            all_first_entries = []
            
            for sensor_num in range(1, 17):
                sensor_col = f'Sens{sensor_num}'
                if sensor_col not in data.columns:
                    continue
                
                voltage = data[sensor_col].values
                
                if SENSOR_DATA and sensor_num in SENSOR_DATA:
                    activation_threshold = SENSOR_DATA[sensor_num].get('activation_threshold', THRESHOLD)
                else:
                    activation_threshold = THRESHOLD
                
                all_critical_times, first_entry, last_exit = find_critical_activation_times(
                    time, voltage, activation_threshold
                )
                
                if len(all_critical_times) > 0:
                    sensor_times_all[sensor_num] = all_critical_times
                    if first_entry is not None:
                        all_first_entries.append(first_entry)
            
            if not all_first_entries:
                print(f"[WARNING] No sensor activations in {csv_file}")
                continue
            
            # Experiment start time (1 second before first sensor trigger)
            experiment_start_time = min(all_first_entries)
            t_zero = experiment_start_time - 1.0
            
            # Collect all activation data
            all_activations = []
            for sensor_num in range(1, 17):
                if sensor_num in sensor_times_all and len(sensor_times_all[sensor_num]) > 0:
                    times = sensor_times_all[sensor_num]
                    if SENSOR_DATA and sensor_num in SENSOR_DATA:
                        position = SENSOR_DATA[sensor_num]['position_m']
                        is_ramp = SENSOR_DATA[sensor_num]['is_ramp']
                    else:
                        position = sensor_num * SENSOR_SPACING
                        is_ramp = sensor_num not in FLAT_REGION_SENSORS
                    
                    for t in times:
                        all_activations.append((t, sensor_num, position, is_ramp))
            
            all_activations.sort(key=lambda x: x[0])
            
            if len(all_activations) == 0:
                print(f"[WARNING] No activations in {csv_file}")
                continue
            
            # Extract sequences
            times_seq = np.array([a[0] for a in all_activations])
            sensors_seq = np.array([a[1] for a in all_activations])
            positions_seq = np.array([a[2] for a in all_activations])
            is_ramp_seq = np.array([a[3] for a in all_activations])
            
            # Adjust times to t=0 reference
            times_seq = times_seq - t_zero
            
            # Calculate speeds (same logic as plot_full_experiment)
            speeds = []
            speed_times = []
            speed_is_ramp = []
            
            i = 0
            while i < len(times_seq):
                current_sensor = sensors_seq[i]
                current_time = times_seq[i]
                current_pos = positions_seq[i]
                current_is_ramp = is_ramp_seq[i]
                
                if i == 0:
                    if len(times_seq) > 1:
                        dt = times_seq[1] - times_seq[0]
                        dx = positions_seq[1] - positions_seq[0]
                        if dt > 0:
                            speed = abs(dx / dt)
                            speeds.append(speed)
                            speed_times.append(current_time)
                            speed_is_ramp.append(current_is_ramp)
                    i += 1
                    
                elif i == len(times_seq) - 1:
                    dt = times_seq[i] - times_seq[i-1]
                    dx = positions_seq[i] - positions_seq[i-1]
                    if dt > 0:
                        speed = abs(dx / dt)
                        speeds.append(speed)
                        speed_times.append(current_time)
                        speed_is_ramp.append(current_is_ramp)
                    i += 1
                    
                elif i + 1 < len(times_seq) and sensors_seq[i + 1] == current_sensor:
                    # Oscillation case
                    next_time = times_seq[i + 1]
                    avg_time = (current_time + next_time) / 2.0
                    
                    if i > 0:
                        dt_pre = avg_time - times_seq[i-1]
                        dx_pre = current_pos - positions_seq[i-1]
                        speed_pre = abs(dx_pre / dt_pre) if dt_pre > 0 else 0
                    else:
                        speed_pre = 0
                    
                    if i + 2 < len(times_seq):
                        dt_post = times_seq[i+2] - avg_time
                        dx_post = positions_seq[i+2] - current_pos
                        speed_post = abs(dx_post / dt_post) if dt_post > 0 else 0
                    else:
                        speed_post = 0
                    
                    if speed_pre > 0 and speed_post > 0:
                        speed = (speed_pre + speed_post) / 2.0
                    elif speed_pre > 0:
                        speed = speed_pre
                    elif speed_post > 0:
                        speed = speed_post
                    else:
                        speed = 0
                    
                    if speed > 0:
                        speeds.append(speed)
                        speed_times.append(avg_time)
                        speed_is_ramp.append(current_is_ramp)
                    
                    i += 2
                    
                else:
                    dt_pre = current_time - times_seq[i-1]
                    dx_pre = current_pos - positions_seq[i-1]
                    speed_pre = abs(dx_pre / dt_pre) if dt_pre > 0 else 0
                    
                    dt_post = times_seq[i+1] - current_time
                    dx_post = positions_seq[i+1] - current_pos
                    speed_post = abs(dx_post / dt_post) if dt_post > 0 else 0
                    
                    if speed_pre > 0 and speed_post > 0:
                        speed = (speed_pre + speed_post) / 2.0
                    elif speed_pre > 0:
                        speed = speed_pre
                    elif speed_post > 0:
                        speed = speed_post
                    else:
                        speed = 0
                    
                    if speed > 0:
                        speeds.append(speed)
                        speed_times.append(current_time)
                        speed_is_ramp.append(current_is_ramp)
                    
                    i += 1
            
            speeds = np.array(speeds)
            speed_times = np.array(speed_times)
            speed_is_ramp = np.array(speed_is_ramp)
            
            # Calculate accelerations from speed fit
            flat_mask_speed = ~speed_is_ramp
            accel_from_fit = None
            accel_from_fit_times = None
            
            if np.any(flat_mask_speed) and np.sum(flat_mask_speed) > 2:
                speeds_flat_fit = speeds[flat_mask_speed]
                times_flat_fit = speed_times[flat_mask_speed]
                speed_poly_coeffs = np.polyfit(times_flat_fit, speeds_flat_fit, 2)
                
                accel_poly_coeffs = np.polyder(speed_poly_coeffs)
                
                accel_from_fit = []
                accel_from_fit_times = []
                
                for i in range(len(speed_times)):
                    accel_at_time = np.polyval(accel_poly_coeffs, speed_times[i])
                    accel_from_fit.append(accel_at_time)
                    accel_from_fit_times.append(speed_times[i])
                
                accel_from_fit = np.array(accel_from_fit)
                accel_from_fit_times = np.array(accel_from_fit_times)
            
            # Store data for this run
            run_data = {
                'filename': csv_path.name,
                'times_seq': times_seq,
                'positions_seq': positions_seq,
                'is_ramp_seq': is_ramp_seq,
                'speed_times': speed_times,
                'speeds': speeds,
                'speed_is_ramp': speed_is_ramp,
                'accel_from_fit_times': accel_from_fit_times,
                'accel_from_fit': accel_from_fit
            }
            
            all_runs_data.append(run_data)
            print(f"[OK] Processed {csv_path.name}")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {csv_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_runs_data) == 0:
        print("[ERROR] No runs successfully processed")
        return False
    
    print(f"\n[INFO] Successfully processed {len(all_runs_data)} runs")
    print(f"[INFO] Generating aggregate plot...")
    
    # Create the aggregate plot
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    
    # Color map for different runs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Collect all flat speed data and all acceleration data for aggregate fitting
    all_flat_speed_times = []
    all_flat_speeds = []
    all_accel_times = []
    all_accels = []
    
    # Plot each run
    for idx, run_data in enumerate(all_runs_data):
        color = colors[idx % 10]
        run_label = f"Run {idx + 1}"
        
        # Position plot
        ax1 = axes[0]
        ax1.scatter(run_data['times_seq'], run_data['positions_seq'],
                   s=40, c=[color], alpha=0.6, marker='o',
                   edgecolors='black', linewidth=0.5, label=run_label, zorder=3)
        
        # Velocity plot
        ax2 = axes[1]
        ax2.scatter(run_data['speed_times'], run_data['speeds'],
                   s=40, c=[color], alpha=0.6, marker='s',
                   edgecolors='black', linewidth=0.5, label=run_label, zorder=3)
        
        # Collect flat region speed data for aggregate fitting
        flat_mask = ~run_data['speed_is_ramp']
        if np.any(flat_mask):
            all_flat_speed_times.extend(run_data['speed_times'][flat_mask].tolist())
            all_flat_speeds.extend(run_data['speeds'][flat_mask].tolist())
        
        # Acceleration plot
        ax3 = axes[2]
        if run_data['accel_from_fit'] is not None and len(run_data['accel_from_fit']) > 0:
            ax3.scatter(run_data['accel_from_fit_times'], run_data['accel_from_fit'],
                       s=40, c=[color], alpha=0.6, marker='o',
                       edgecolors='black', linewidth=0.5, label=run_label, zorder=3)
            
            # Collect all acceleration data for aggregate statistics
            all_accel_times.extend(run_data['accel_from_fit_times'].tolist())
            all_accels.extend(run_data['accel_from_fit'].tolist())
    
    # Convert to numpy arrays
    all_flat_speed_times = np.array(all_flat_speed_times)
    all_flat_speeds = np.array(all_flat_speeds)
    all_accel_times = np.array(all_accel_times)
    all_accels = np.array(all_accels)
    
    # Calculate the overall average acceleration across all runs
    # This should match the average of individual run accelerations
    avg_acceleration_overall = np.mean(all_accels)
    print(f"[INFO] Average acceleration (all points): {avg_acceleration_overall:.6f} m/s²")
    
    # Configure position plot
    ax1 = axes[0]
    ax1.set_ylabel('Position (m)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{experiment_name} - Aggregate Kinematics (All Runs)', 
                  fontsize=16, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=12)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='t=0')
    
    # Configure velocity plot
    ax2 = axes[1]
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_ylabel('Velocity (m/s)', fontsize=14, fontweight='bold')
    
    # Add aggregate speed fit line (quadratic fit through all flat region points)
    if len(all_flat_speed_times) > 2:
        # Fit quadratic polynomial to all flat speed data
        speed_poly_coeffs = np.polyfit(all_flat_speed_times, all_flat_speeds, 2)
        
        # Generate fitted curve
        t_min = all_flat_speed_times.min()
        t_max = all_flat_speed_times.max()
        t_fit = np.linspace(t_min, t_max, 500)
        speed_fit = np.polyval(speed_poly_coeffs, t_fit)
        
        ax2.plot(t_fit, speed_fit, 'b-', linewidth=2.5, 
                label='Aggregate fit (flat regions)', zorder=10, alpha=0.8)
        print(f"[INFO] Velocity fit coefficients (quadratic): {speed_poly_coeffs}")
    
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=12)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Configure acceleration plot
    ax3 = axes[2]
    ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_ylabel('Acceleration (m/s²)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    
    # Calculate average acceleration and add fitted line through all points
    if len(all_accels) > 0:
        avg_acceleration = np.mean(all_accels)
        
        # Fit linear line through all acceleration points
        if len(all_accel_times) > 1:
            accel_poly_coeffs = np.polyfit(all_accel_times, all_accels, 1)
            
            # Generate fitted line
            t_min_accel = all_accel_times.min()
            t_max_accel = all_accel_times.max()
            t_accel_fit = np.linspace(t_min_accel, t_max_accel, 500)
            accel_fit = np.polyval(accel_poly_coeffs, t_accel_fit)
            
            ax3.plot(t_accel_fit, accel_fit, 'r-', linewidth=2.5, 
                    label=f'Aggregate fit (Avg: {avg_acceleration:.6f} m/s²)', 
                    zorder=10, alpha=0.8)
            print(f"[INFO] Average acceleration: {avg_acceleration:.6f} m/s²")
            print(f"[INFO] Acceleration fit coefficients (linear): {accel_poly_coeffs}")
    
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=12)
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Aggregate plot saved: {output_path}")
    
    plt.close(fig)
    
    return True


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        graph_on_file(sys.argv[1])
    else:
        print("Usage: python graph.py <csv_file_path>")
        print("Example: python graph.py data/No_Tyre_Control/csv/NoTyre_1.csv")
