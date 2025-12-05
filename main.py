from pathlib import Path
import csv
import re
from dataprocess import graph_on_file
from dataprocess.graph import plot_aggregate_runs

CSV_FILE = "data/No_Tyre_Control/csv/NoTyre_1.csv"
SAVE_DIR = "output/graphs"
SAVE_GRAPHS = True

def process_single_file(csv_file, save_dir=None, save_graphs=True):
    """
    Process a single CSV file and generate kinematics graphs.
    
    Args:
        csv_file: Path to CSV file
        save_dir: Output directory (None uses CSV directory)
        save_graphs: Whether to save graphs
    
    Returns:
        Dictionary with metrics, or None if failed
    """
    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"\n[ERROR] CSV file not found: {csv_file}")
        return None
    
    if not save_graphs:
        save_param = False
    elif save_dir:
        save_param = save_dir
    else:
        save_param = True
    
    try:
        metrics = graph_on_file(
            csv_file_path=csv_file,
            save=save_param
        )
        
        if metrics and save_graphs:
            save_location = save_dir if save_dir else csv_path.parent
            print(f"\n[SAVED] Graphs saved to: {save_location}")
            print(f"        - {csv_path.stem}_voltage_time.png")
            print(f"        - {csv_path.stem}_posvelacc_time.png")
        
        return metrics
            
    except Exception as e:
        print(f"\n[ERROR] Failed to process {csv_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run single file analysis pipeline."""
    print("="*80)
    print(" " * 20 + "Wheel Rolling Experiment Analysis")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  CSV File:    {CSV_FILE}")
    print(f"  Save Dir:    {SAVE_DIR if SAVE_DIR else '(same as CSV)'}")
    print(f"  Save Graphs: {SAVE_GRAPHS}")
    print("\n" + "-"*80)
    
    metrics = process_single_file(CSV_FILE, SAVE_DIR, SAVE_GRAPHS)
    
    if metrics:
        print("\n" + "="*80)
        print(" " * 30 + "Analysis Complete!")
        print("="*80)
        print(f"\nYou can modify the CSV_FILE and SAVE_DIR variables")
        print(f"in main.py to process different data files.")
    else:
        print("\n[ERROR] Analysis failed. Check the error messages above.")
    
    print("\n" + "="*80)


def parse_filename_and_output_dir(csv_path, base_output_dir):
    """
    Parse CSV filename and determine output directory structure.
    
    Examples:
        NoPattern_5mm_1.csv -> result/plots/no_pattern/5mm/1
        Directional_7mm_2.csv -> result/plots/directional/7mm/2
        NoTyre_1.csv -> result/plots/no_tyre_control/1
    
    Returns:
        Tuple of (run_id, output_dir) or (None, None) if parsing fails
    """
    filename = csv_path.stem  # filename without extension
    
    # Pattern for files with thickness: Pattern_Xmm_Y
    pattern_with_thickness = r'^(NoPattern|Directional|Symmetrical)_(\d+)mm_(\d+)$'
    # Pattern for NoTyre files: NoTyre_Y
    pattern_no_tyre = r'^NoTyre_(\d+)$'
    
    match_with_thickness = re.match(pattern_with_thickness, filename)
    match_no_tyre = re.match(pattern_no_tyre, filename)
    
    if match_with_thickness:
        pattern_type = match_with_thickness.group(1)
        thickness_mm = match_with_thickness.group(2)
        run_number = match_with_thickness.group(3)
        
        # Convert pattern type to lowercase with underscores
        pattern_dir = pattern_type.lower()
        if pattern_type == "NoPattern":
            pattern_dir = "no_pattern"
        
        output_dir = Path(base_output_dir) / pattern_dir / f"{thickness_mm}mm" / run_number
        run_id = f"{pattern_type}_{thickness_mm}mm_{run_number}"
        
        return run_id, output_dir
    
    elif match_no_tyre:
        run_number = match_no_tyre.group(1)
        output_dir = Path(base_output_dir) / "no_tyre_control" / run_number
        run_id = f"NoTyre_{run_number}"
        
        return run_id, output_dir
    
    else:
        print(f"[WARNING] Could not parse filename: {filename}")
        return None, None


def update_result_aggregate(csv_path, run_id, metrics):
    """
    Update result_aggregate.csv with duration and acceleration values.
    If run_id doesn't exist, a new row will be added.
    
    Args:
        csv_path: Path to result_aggregate.csv
        run_id: Run identifier (e.g., "NoTyre_1", "NoPattern_5mm_1")
        metrics: Dictionary with 'duration_s' and 'acceleration_ms2'
    """
    if not csv_path.exists():
        print(f"[ERROR] Result aggregate file not found: {csv_path}")
        return False
    
    if metrics is None or 'duration_s' not in metrics or 'acceleration_ms2' not in metrics:
        print(f"[WARNING] Missing metrics for {run_id}")
        return False
    
    # Parse run_id to extract components
    # Format: NoTyre_1 or NoPattern_5mm_1 or Directional_7mm_2, etc.
    if run_id.startswith("NoTyre_"):
        parts = run_id.split("_")
        experiment_type = "control"
        pattern_type = "no_tyre"
        thickness_mm = ""
        run_number = parts[1]
    else:
        # Format: Pattern_Xmm_Y
        parts = run_id.split("_")
        pattern_type = parts[0].lower()
        if pattern_type == "nopattern":
            pattern_type = "no_pattern"
        thickness_mm = parts[1].replace("mm", "")
        run_number = parts[2]
        experiment_type = "experimental"
    
    # Read all rows
    rows = []
    row_found = False
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = [f for f in reader.fieldnames if f is not None]  # Filter out None fields
        for row in reader:
            # Remove None key if present (from trailing commas)
            if None in row:
                del row[None]
            
            if row['run_id'] == run_id:
                # Update this row
                row['acceleration_ms2'] = f"{metrics['acceleration_ms2']:.6f}"
                row['duration_s'] = f"{metrics['duration_s']:.6f}"
                row_found = True
            rows.append(row)
    
    # If row not found, add a new one
    if not row_found:
        new_row = {
            'run_id': run_id,
            'experiment_type': experiment_type,
            'pattern_type': pattern_type,
            'thickness_mm': thickness_mm,
            'run_number': run_number,
            'acceleration_ms2': f"{metrics['acceleration_ms2']:.6f}",
            'duration_s': f"{metrics['duration_s']:.6f}",
            'tyre_weight_g': '',
            'assembly_weight_g': '',
            'wheel_radius_cm': ''
        }
        rows.append(new_row)
    
    # Write back all rows
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    return True


def populate_acceleration_data():
    """
    Process all experimental CSV files to calculate acceleration WITHOUT saving graphs.
    Updates result_aggregate.csv with duration and acceleration data.
    """
    # Define input directories
    data_dirs = [
        Path("data/Directional_Tyre/csv"),
        Path("data/No_Pattern/csv"),
        Path("data/No_Tyre_Control/csv"),
        Path("data/Symmetrical_Tyre/csv")
    ]
    
    result_aggregate_path = Path("result/result_aggregate.csv")
    
    print("="*80)
    print(" " * 15 + "Populating Acceleration Data (No Graph Saving)")
    print("="*80)
    
    # Collect all CSV files
    all_csv_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            csv_files = sorted(data_dir.glob("*.csv"))
            all_csv_files.extend(csv_files)
            print(f"Found {len(csv_files)} files in {data_dir}")
        else:
            print(f"[WARNING] Directory not found: {data_dir}")
    
    print(f"\nTotal files to process: {len(all_csv_files)}")
    print("\n" + "-"*80)
    
    # Process each file
    results = []
    for i, csv_file in enumerate(all_csv_files, 1):
        print(f"\n[{i}/{len(all_csv_files)}] Processing: {csv_file.name}")
        
        # Parse filename to get run_id
        run_id, _ = parse_filename_and_output_dir(csv_file, Path("result/plots"))
        
        if run_id is None:
            print(f"[SKIP] Could not parse filename: {csv_file.name}")
            results.append((csv_file.name, None, "Parse Error"))
            continue
        
        # Process file WITHOUT saving graphs
        try:
            metrics = graph_on_file(str(csv_file), output_dir=None, save=False)
            
            if metrics is not None:
                # Update result_aggregate.csv
                success = update_result_aggregate(result_aggregate_path, run_id, metrics)
                if success:
                    results.append((csv_file.name, metrics, "Success"))
                    print(f"[OK] Updated result_aggregate.csv for {run_id}")
                else:
                    results.append((csv_file.name, metrics, "CSV Update Failed"))
            else:
                results.append((csv_file.name, None, "Processing Failed"))
        
        except Exception as e:
            print(f"[ERROR] Exception occurred: {str(e)}")
            results.append((csv_file.name, None, f"Error: {str(e)}"))
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "Processing Summary")
    print("="*80)
    
    successful = sum(1 for _, _, status in results if status == "Success")
    failed = len(results) - successful
    
    print(f"\nTotal files:  {len(results)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for filename, _, status in results:
            if status != "Success":
                print(f"  - {filename}: {status}")
    
    print("\n" + "="*80)
    print(f"Updated CSV: {result_aggregate_path}")
    print("="*80)


def batch_process_all_experiments():
    """
    Process all experimental CSV files and generate graphs.
    Updates result_aggregate.csv with duration and acceleration data.
    
    This function:
    1. Finds all CSV files in data directories
    2. Processes each file and generates graphs
    3. Updates result_aggregate.csv with duration and acceleration
    """
    # Define input directories
    data_dirs = [
        Path("data/Directional_Tyre/csv"),
        Path("data/No_Pattern/csv"),
        Path("data/No_Tyre_Control/csv"),
        Path("data/Symmetrical_Tyre/csv")
    ]
    
    # Output base directory for plots
    output_base = Path("result/plots")
    result_aggregate_path = Path("result/result_aggregate.csv")
    
    print("="*80)
    print(" " * 20 + "Batch Processing All Experiments")
    print("="*80)
    
    # Collect all CSV files
    all_csv_files = []
    for data_dir in data_dirs:
        if data_dir.exists():
            csv_files = sorted(data_dir.glob("*.csv"))
            all_csv_files.extend(csv_files)
            print(f"Found {len(csv_files)} files in {data_dir}")
        else:
            print(f"[WARNING] Directory not found: {data_dir}")
    
    print(f"\nTotal files to process: {len(all_csv_files)}")
    print("\n" + "-"*80)
    
    # Process each file
    results = []
    for i, csv_file in enumerate(all_csv_files, 1):
        print(f"\n[{i}/{len(all_csv_files)}] Processing: {csv_file.name}")
        
        # Parse filename and determine output directory
        run_id, output_dir = parse_filename_and_output_dir(csv_file, output_base)
        
        if run_id is None:
            print(f"[SKIP] Could not parse filename: {csv_file.name}")
            results.append((csv_file.name, None, "Parse Error"))
            continue
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process file
        try:
            metrics = graph_on_file(str(csv_file), output_dir=str(output_dir), save=True)
            
            if metrics is not None:
                # Update result_aggregate.csv
                success = update_result_aggregate(result_aggregate_path, run_id, metrics)
                if success:
                    results.append((csv_file.name, metrics, "Success"))
                    print(f"[OK] Updated result_aggregate.csv for {run_id}")
                else:
                    results.append((csv_file.name, metrics, "CSV Update Failed"))
            else:
                results.append((csv_file.name, None, "Processing Failed"))
        
        except Exception as e:
            print(f"[ERROR] Exception occurred: {str(e)}")
            results.append((csv_file.name, None, f"Error: {str(e)}"))
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "Processing Summary")
    print("="*80)
    
    successful = sum(1 for _, _, status in results if status == "Success")
    failed = len(results) - successful
    
    print(f"\nTotal files:  {len(results)}")
    print(f"Successful:   {successful}")
    print(f"Failed:       {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for filename, _, status in results:
            if status != "Success":
                print(f"  - {filename}: {status}")
    
    print("\n" + "="*80)
    print(f"\nResults saved to: {output_base}")
    print(f"Updated CSV: {result_aggregate_path}")
    print("="*80)


def batch_generate_aggregate_plots():
    """
    Generate aggregate plots for all experiment groups.
    
    This function creates position, velocity, and acceleration plots
    for all 5 runs of each experiment group overlaid on the same graph.
    
    Experiment groups:
    - Directional: 5mm, 7mm, 9mm
    - Symmetrical: 5mm, 7mm, 9mm
    - NoPattern: 5mm, 7mm, 9mm
    - NoTyre: control (no thickness variations)
    """
    print("="*80)
    print(" " * 15 + "Batch Generate Aggregate Plots for All Experiment Groups")
    print("="*80)
    
    # Define all experiment groups
    experiment_groups = [
        # Directional experiments
        {
            'name': 'Directional 5mm',
            'csv_files': [f"data/Directional_Tyre/csv/Directional_5mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/directional/Directional_5mm_aggregate.png'
        },
        {
            'name': 'Directional 7mm',
            'csv_files': [f"data/Directional_Tyre/csv/Directional_7mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/directional/Directional_7mm_aggregate.png'
        },
        {
            'name': 'Directional 9mm',
            'csv_files': [f"data/Directional_Tyre/csv/Directional_9mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/directional/Directional_9mm_aggregate.png'
        },
        # Symmetrical experiments
        {
            'name': 'Symmetrical 5mm',
            'csv_files': [f"data/Symmetrical_Tyre/csv/Symmetrical_5mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/symmetrical/Symmetrical_5mm_aggregate.png'
        },
        {
            'name': 'Symmetrical 7mm',
            'csv_files': [f"data/Symmetrical_Tyre/csv/Symmetrical_7mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/symmetrical/Symmetrical_7mm_aggregate.png'
        },
        {
            'name': 'Symmetrical 9mm',
            'csv_files': [f"data/Symmetrical_Tyre/csv/Symmetrical_9mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/symmetrical/Symmetrical_9mm_aggregate.png'
        },
        # No Pattern experiments
        {
            'name': 'No Pattern 5mm',
            'csv_files': [f"data/No_Pattern/csv/NoPattern_5mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/no_pattern/NoPattern_5mm_aggregate.png'
        },
        {
            'name': 'No Pattern 7mm',
            'csv_files': [f"data/No_Pattern/csv/NoPattern_7mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/no_pattern/NoPattern_7mm_aggregate.png'
        },
        {
            'name': 'No Pattern 9mm',
            'csv_files': [f"data/No_Pattern/csv/NoPattern_9mm_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/no_pattern/NoPattern_9mm_aggregate.png'
        },
        # No Tyre control
        {
            'name': 'No Tyre (Control)',
            'csv_files': [f"data/No_Tyre_Control/csv/NoTyre_{i}.csv" for i in range(1, 6)],
            'output_path': 'result/plots/no_tyre_control/NoTyre_aggregate.png'
        }
    ]
    
    print(f"\nTotal experiment groups: {len(experiment_groups)}")
    print("\n" + "-"*80)
    
    # Process each experiment group
    results = []
    for i, group in enumerate(experiment_groups, 1):
        print(f"\n[{i}/{len(experiment_groups)}] Processing: {group['name']}")
        print(f"Output: {group['output_path']}")
        
        try:
            success = plot_aggregate_runs(
                csv_files=group['csv_files'],
                output_path=group['output_path'],
                experiment_name=group['name']
            )
            
            if success:
                results.append((group['name'], 'Success'))
                print(f"[OK] Successfully generated aggregate plot for {group['name']}")
            else:
                results.append((group['name'], 'Failed'))
                print(f"[ERROR] Failed to generate aggregate plot for {group['name']}")
        
        except Exception as e:
            results.append((group['name'], f'Error: {str(e)}'))
            print(f"[ERROR] Exception occurred for {group['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    print("\n" + "="*80)
    print(" " * 25 + "Processing Summary")
    print("="*80)
    
    successful = sum(1 for _, status in results if status == 'Success')
    failed = len(results) - successful
    
    print(f"\nTotal groups:  {len(results)}")
    print(f"Successful:    {successful}")
    print(f"Failed:        {failed}")
    
    if failed > 0:
        print("\nFailed groups:")
        for name, status in results:
            if status != 'Success':
                print(f"  - {name}: {status}")
    
    print("\n" + "="*80)
    print(f"\nAggregate plots saved to: result/plots/")
    print("  - result/plots/directional/")
    print("  - result/plots/symmetrical/")
    print("  - result/plots/no_pattern/")
    print("  - result/plots/no_tyre_control/")
    print("="*80)


if __name__ == "__main__":
    # Single file mode
    # main()
    
    # Populate acceleration data without saving graphs (no plots)
    # populate_acceleration_data()
    
    # ============================================================================
    # FULL REGENERATION: All individual plots AND aggregate plots
    # ============================================================================
    
    print("\n" + "="*80)
    print(" " * 20 + "FULL PLOT REGENERATION PIPELINE")
    print("="*80)
    print("\nThis will regenerate:")
    print("  1. All individual voltage time plots")
    print("  2. All individual position/velocity/acceleration plots")
    print("  3. All aggregate plots for experiment groups")
    print("\nTotal files to process: ~50 individual runs + 10 aggregate groups")
    print("="*80 + "\n")
    
    # Step 1: Generate all individual run plots
    print("\n" + "="*80)
    print(" " * 25 + "STEP 1: Individual Runs")
    print("="*80)
    batch_process_all_experiments()
    
    # Step 2: Generate all aggregate plots
    print("\n" + "="*80)
    print(" " * 25 + "STEP 2: Aggregate Plots")
    print("="*80)
    batch_generate_aggregate_plots()
    
    # Final summary
    print("\n" + "="*80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("="*80)
    print("\nAll plots have been regenerated successfully.")
    print("Check result/plots/ for all output files.")
    print("="*80 + "\n")
