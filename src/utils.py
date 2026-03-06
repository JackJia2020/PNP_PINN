import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import tensorflow as tf

def plot_model_comparison(
    pinn_path="./outputs/PNP/outputs_PNP/inferencers/inf_end.npz",
    nn_path="./outputs/PNP_supervised/outputs_PNP/inferencers/inf_end.npz",
    fipy_path="./data/ground_truth.csv",
    save_path=None
):
    """
    Loads inferencer data and FDM ground truth, interpolates to a common grid,
    and generates a 5-column contour plot comparing the models.
    """
    print("Loading data for comparison plot...")
    
    # 1. Load Data
    raw_data_pinn = np.load(pinn_path, allow_pickle=True)
    pinn_dict = raw_data_pinn['arr_0'].item()

    raw_data_nn = np.load(nn_path, allow_pickle=True)
    nn_dict = raw_data_nn['arr_0'].item()

    fipy_df = pd.read_csv(fipy_path)
    fipy_pts = fipy_df[['x', 't']].values

    # Prepare Grid 
    X_grid = pinn_dict['x'].reshape(400, 100)
    T_grid = pinn_dict['t'].reshape(400, 100)

    # 2. Initialize Plot
    fig, axes = plt.subplots(3, 5, figsize=(24, 12), sharex=True, sharey=True)
    variables = ['cp', 'cn', 'phi']
    titles = ['Cations ($c_p$)', 'Anions ($c_n$)', 'Potential ($\phi$)']

    for i, var in enumerate(variables):
        print(f"Processing {var}...")
        data_pinn = pinn_dict[var].reshape(400, 100)
        data_nn = nn_dict[var].reshape(400, 100)
        data_fipy = griddata(fipy_pts, fipy_df[var].values, (X_grid, T_grid), method='linear')

        error_pinn = np.abs(data_fipy - data_pinn)
        error_nn = np.abs(data_fipy - data_nn)

        vmin_pred = np.nanmin([data_fipy, data_pinn, data_nn])
        vmax_pred = np.nanmax([data_fipy, data_pinn, data_nn])
        vmax_err = np.nanmax([error_pinn, error_nn])

        # Col 0: Truth
        im0 = axes[i, 0].contourf(X_grid, T_grid, data_fipy, levels=50, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        axes[i, 0].set_title(f'FDM Truth: {titles[i]}')
        fig.colorbar(im0, ax=axes[i, 0])

        # Col 1: PINN Pred
        im1 = axes[i, 1].contourf(X_grid, T_grid, data_pinn, levels=50, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        axes[i, 1].set_title(f'PINN Pred: {titles[i]}')
        fig.colorbar(im1, ax=axes[i, 1])

        # Col 2: NN Pred
        im2 = axes[i, 2].contourf(X_grid, T_grid, data_nn, levels=50, cmap='viridis', vmin=vmin_pred, vmax=vmax_pred)
        axes[i, 2].set_title(f'NN Pred: {titles[i]}')
        fig.colorbar(im2, ax=axes[i, 2])

        # Col 3: PINN Error
        im3 = axes[i, 3].contourf(X_grid, T_grid, error_pinn, levels=50, cmap='magma', vmin=0, vmax=vmax_err)
        axes[i, 3].set_title('PINN Abs Error')
        fig.colorbar(im3, ax=axes[i, 3])

        # Col 4: NN Error
        im4 = axes[i, 4].contourf(X_grid, T_grid, error_nn, levels=50, cmap='magma', vmin=0, vmax=vmax_err)
        axes[i, 4].set_title('NN Abs Error')
        fig.colorbar(im4, ax=axes[i, 4])

    for ax in axes[-1, :]: ax.set_xlabel('Position (x)')
    for ax in axes[:, 0]:  ax.set_ylabel('Time (t)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def track_and_plot_loss(
    log_dir="./outputs/PNP/outputs_PNP", 
    csv_save_path="./outputs/merged_loss_history.csv",
    plot_save_path=None
):
    """
    Extracts loss metrics from TensorFlow event files, deduplicates them, 
    saves to CSV, and plots the loss history on a log scale.
    """
    log_files = sorted(glob.glob(f"{log_dir}/events.out.tfevents.*"))

    if not log_files:
        print(f"No log files found in {log_dir}. Check your directory path.")
        return

    print(f"Found {len(log_files)} log files. Extracting and merging...")
    records = []
    
    for log_file in log_files:
        try:
            for e in tf.compat.v1.train.summary_iterator(log_file):
                for v in e.summary.value:
                    if v.tag.startswith('Train/') and 'learning_rate' not in v.tag and 'time' not in v.tag and 'step' not in v.tag:
                        if v.HasField('tensor'):
                            val = tf.make_ndarray(v.tensor).item()
                        elif v.HasField('simple_value'):
                            val = v.simple_value
                        else:
                            continue
                        records.append({'Step': e.step, 'Constraint': v.tag, 'Loss_Value': val})
        except Exception as err:
            print(f"Skipped a corrupted or incomplete record in {log_file}: {err}")

    df = pd.DataFrame(records)

    if not df.empty:
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Step', 'Constraint'], keep='last')
        dropped = initial_count - len(df)
        print(f"Merged successfully. Removed {dropped} overlapping duplicate records.")

        df_pivot = df.pivot_table(index='Step', columns='Constraint', values='Loss_Value')
        
        # Ensure the directory exists before saving
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        df_pivot.to_csv(csv_save_path)
        print(f"Saved complete history to '{csv_save_path}'\n")

        plt.figure(figsize=(14, 8))
        for column in df_pivot.columns:
            clean_label = column.replace('Train/', '').replace('loss_', '')
            if column in ['Train/loss', 'Train/loss_aggregated']:
                plt.plot(df_pivot.index, df_pivot[column], label='Total Loss', color='black', linewidth=3)
            elif any(keyword in column for keyword in ['poisson', 'nernst', 'pde']):
                plt.plot(df_pivot.index, df_pivot[column], label=f"PDE: {clean_label}", linestyle='--', linewidth=2)
            else:
                plt.plot(df_pivot.index, df_pivot[column], label=f"BC/IC: {clean_label}", alpha=0.5, linestyle=':')

        plt.yscale('log')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss Value (Log Scale)')
        plt.title('Complete PINN Loss History (Merged Logs)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if plot_save_path:
            plt.savefig(plot_save_path)
        plt.show()
    else:
        print("No loss data found in any of the files.")

# Allow the script to be run directly from the command line
if __name__ == "__main__":
    print("This is a utility script. Import these functions into your notebook or run them below.")
    # You can uncomment these to test them directly:
    # plot_model_comparison()
    # track_and_plot_loss()

def evaluate_and_plot_snapshot(
    pinn_path="./outputs/PNP/outputs_PNP/inferencers/inf_end.npz",
    nn_path="./outputs/PNP_supervised/outputs_PNP/inferencers/inf_end.npz",
    fipy_path="./data/ground_truth.csv",
    t_snap=0.2,
    save_path=None
):
    """
    Calculates RMSE and Relative L2 errors for both PINN and NN against the 
    FDM benchmark. Plots a 1D cross-section of the concentration at t_snap.
    """
    print(f"\n--- Quantitative Evaluation at t={t_snap} ---")
    
    # 1. Load Data
    raw_pinn = np.load(pinn_path, allow_pickle=True)['arr_0'].item()
    raw_nn = np.load(nn_path, allow_pickle=True)['arr_0'].item()
    fipy_df = pd.read_csv(fipy_path)
    
    # We use griddata to map FiPy's (x, t) values onto the continuous grid
    fipy_points = fipy_df[['x', 't']].values
    
    # 2. Helper function to calculate and print metrics
    def calc_metrics(model_dict, model_name):
        df = pd.DataFrame({
            'x': model_dict['x'].flatten(),
            't': model_dict['t'].flatten(),
            'cp_pred': model_dict['cp'].flatten(),
            'cn_pred': model_dict['cn'].flatten(),
            'phi_pred': model_dict['phi'].flatten(),
            'cp_true': griddata(fipy_points, fipy_df['cp'].values, (model_dict['x'], model_dict['t']), method='linear').flatten(),
            'cn_true': griddata(fipy_points, fipy_df['cn'].values, (model_dict['x'], model_dict['t']), method='linear').flatten(),
            'phi_true': griddata(fipy_points, fipy_df['phi'].values, (model_dict['x'], model_dict['t']), method='linear').flatten()
        }).dropna()
        
        print(f"\n[{model_name} Error Metrics]")
        for var in ['cp', 'cn', 'phi']:
            rmse = np.sqrt(np.mean((df[f'{var}_pred'] - df[f'{var}_true'])**2))
            rel_l2 = np.linalg.norm(df[f'{var}_pred'] - df[f'{var}_true']) / np.linalg.norm(df[f'{var}_true'])
            print(f"{var.ljust(3)} - RMSE: {rmse:.6f} | Relative L2: {rel_l2:.6f}")
            
        return df

    # Calculate metrics for both models
    df_pinn = calc_metrics(raw_pinn, "PINN")
    df_nn = calc_metrics(raw_nn, "Supervised NN")

    # 3. Visualization Snapshot at t_snap
    # Find the closest time step in the continuous grid to our desired t_snap
    t_unique = np.sort(df_pinn["t"].unique())
    t0 = t_unique[np.argmin(np.abs(t_unique - t_snap))]
    
    sub_pinn = df_pinn[df_pinn["t"] == t0].sort_values("x")
    sub_nn = df_nn[df_nn["t"] == t0].sort_values("x")

    plt.figure(figsize=(12, 7))
    
    # Plot Ground Truth (Solid Lines)
    plt.plot(sub_pinn['x'], sub_pinn['cp_true'], 'b-', linewidth=2, label='FDM Truth ($c_p$)')
    plt.plot(sub_pinn['x'], sub_pinn['cn_true'], 'r-', linewidth=2, label='FDM Truth ($c_n$)')
    
    # Plot PINN (Dashed Lines)
    plt.plot(sub_pinn['x'], sub_pinn['cp_pred'], 'b--', linewidth=2, label='PINN ($c_p$)')
    plt.plot(sub_pinn['x'], sub_pinn['cn_pred'], 'r--', linewidth=2, label='PINN ($c_n$)')
    
    # Plot Supervised NN (Dotted Lines)
    plt.plot(sub_nn['x'], sub_nn['cp_pred'], 'b:', linewidth=3, label='Supervised NN ($c_p$)')
    plt.plot(sub_nn['x'], sub_nn['cn_pred'], 'r:', linewidth=3, label='Supervised NN ($c_n$)')

    plt.title(f'1D Cross-Section Comparison at t = {t_snap}')
    plt.xlabel('Position (x)')
    plt.ylabel('Concentration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
