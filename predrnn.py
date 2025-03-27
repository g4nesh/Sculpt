import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import random
import sys
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

np.random.seed(42)
random.seed(42)

AMINO_ACID_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'volume': 88.6},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'volume': 108.5},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'volume': 111.1},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'volume': 138.4},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'volume': 189.9},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'volume': 60.1},
    'H': {'hydrophobicity': -3.2, 'charge': 1, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'volume': 168.6},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'volume': 166.7},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'volume': 162.9},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'volume': 114.1},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'volume': 112.7},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'volume': 143.8},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'volume': 173.4},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'volume': 89.0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'volume': 116.1},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'volume': 140.0},
    'W': {'hydrophobicity': 0.9, 'charge': 0, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'volume': 193.6},
}

def validate_amino_acid_sequence(sequence):
    amino_acids = set('ACDEFGHIKLMNPQRSTVWY')
    sequence = sequence.strip().upper()
    if not sequence:
        raise ValueError("Amino acid sequence cannot be empty")
    if len(sequence) < 5:
        raise ValueError("Sequence must be at least 5 amino acids long")
    for char in sequence:
        if char not in amino_acids:
            raise ValueError(f"Invalid amino acid '{char}' detected")
    return sequence

def calculate_sequence_properties(sequence):
    hydrophobicity = 0.0
    charge = 0.0
    volume = 0.0
    length = len(sequence)

    for aa in sequence:
        props = AMINO_ACID_PROPERTIES[aa]
        hydrophobicity += props['hydrophobicity']
        charge += props['charge']
        volume += props['volume']

    return {
        'length': length,
        'avg_hydrophobicity': hydrophobicity / length,
        'avg_charge': charge / length,
        'avg_volume': volume / length
    }

def generate_z_scores(sequence_properties, num_points=50):
    logger.info("Generating Z-scores based on sequence properties...")
    age = np.logspace(np.log10(0.04), np.log10(50), num_points)
    base_trend_protein = np.linspace(1.2, -1.2, num_points) + np.random.normal(0, 0.25, num_points)
    base_trend_rna = np.linspace(0.9, -0.7, num_points) + np.random.normal(0, 0.25, num_points)

    length_factor = sequence_properties['length'] / 100.0
    hydrophobicity_factor = sequence_properties['avg_hydrophobicity'] / 5.0
    charge_factor = sequence_properties['avg_charge']
    volume_factor = sequence_properties['avg_volume'] / 200.0

    protein_z_scores = base_trend_protein + (hydrophobicity_factor * np.random.normal(0, 0.6, num_points))
    protein_z_scores += length_factor * np.random.normal(0, 0.4, num_points) + volume_factor * np.random.normal(0, 0.3, num_points)
    rna_z_scores = base_trend_rna + (charge_factor * np.random.normal(0, 0.6, num_points))
    rna_z_scores += length_factor * np.random.normal(0, 0.4, num_points) + volume_factor * np.random.normal(0, 0.3, num_points)

    protein_scatter = protein_z_scores + np.random.normal(0, 0.6, num_points)
    rna_scatter = rna_z_scores + np.random.normal(0, 0.6, num_points)

    protein_scatter = np.clip(protein_scatter, -4.5, 2.5)
    rna_scatter = np.clip(rna_scatter, -4.5, 2.5)

    return age, protein_scatter, rna_scatter, protein_z_scores, rna_z_scores

def fit_trend_line(x, y, num_points=100):
    spline = UnivariateSpline(x, y, s=1.5)
    x_smooth = np.logspace(np.log10(min(x)), np.log10(max(x)), num_points)
    y_smooth = spline(x_smooth)
    return x_smooth, y_smooth

def calculate_confidence_intervals(y, ci_factor=0.6):
    ci = ci_factor * np.std(y)
    return y - ci, y + ci

def find_significant_points(x, y, threshold=0.8):
    peaks = []
    troughs = []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1] and abs(y[i]) > threshold:
            peaks.append((x[i], y[i]))
        if y[i] < y[i - 1] and y[i] < y[i + 1] and abs(y[i]) > threshold:
            troughs.append((x[i], y[i]))
    return peaks, troughs

def create_expression_plot(sequence, gene_name="DCLK1", save_plot=True, output_dir="plots"):
    logger.info(f"Creating expression plot for sequence: {sequence}")
    sequence = validate_amino_acid_sequence(sequence)
    seq_properties = calculate_sequence_properties(sequence)
    age, protein_scatter, rna_scatter, protein_trend, rna_trend = generate_z_scores(seq_properties)

    protein_x_smooth, protein_y_smooth = fit_trend_line(age, protein_trend)
    rna_x_smooth, rna_y_smooth = fit_trend_line(age, rna_trend)

    protein_ci_lower, protein_ci_upper = calculate_confidence_intervals(protein_y_smooth)
    rna_ci_lower, rna_ci_upper = calculate_confidence_intervals(rna_y_smooth)

    protein_peaks, protein_troughs = find_significant_points(age, protein_trend)
    rna_peaks, rna_troughs = find_significant_points(age, rna_trend)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    ax.scatter(age, protein_scatter, c='#1f77b4', marker='D', label='Protein Expression', alpha=0.7, s=60)
    ax.scatter(age, rna_scatter, c='#ff7f0e', marker='^', label='RNA Expression', alpha=0.7, s=60)

    ax.plot(protein_x_smooth, protein_y_smooth, c='#1f77b4', linestyle='-', linewidth=2.5, label='Protein Trend')
    ax.plot(rna_x_smooth, rna_y_smooth, c='#ff7f0e', linestyle='-', linewidth=2.5, label='RNA Trend')

    ax.fill_between(protein_x_smooth, protein_ci_lower, protein_ci_upper, color='#1f77b4', alpha=0.15, label='Protein CI')
    ax.fill_between(rna_x_smooth, rna_ci_lower, rna_ci_upper, color='#ff7f0e', alpha=0.15, label='RNA CI')

    for peak in protein_peaks:
        ax.annotate(f'{peak[1]:.2f}', xy=peak, xytext=(10, 10), textcoords='offset points',
                    color='#1f77b4', fontsize=9, arrowprops=dict(arrowstyle='->', color='#1f77b4'))
    for trough in protein_troughs:
        ax.annotate(f'{trough[1]:.2f}', xy=trough, xytext=(10, -10), textcoords='offset points',
                    color='#1f77b4', fontsize=9, arrowprops=dict(arrowstyle='->', color='#1f77b4'))
    for peak in rna_peaks:
        ax.annotate(f'{peak[1]:.2f}', xy=peak, xytext=(10, 10), textcoords='offset points',
                    color='#ff7f0e', fontsize=9, arrowprops=dict(arrowstyle='->', color='#ff7f0e'))
    for trough in rna_troughs:
        ax.annotate(f'{trough[1]:.2f}', xy=trough, xytext=(10, -10), textcoords='offset points',
                    color='#ff7f0e', fontsize=9, arrowprops=dict(arrowstyle='->', color='#ff7f0e'))

    ax.set_xscale('log')
    ax.set_xlim(0.03, 60)
    ax.set_ylim(-5, 3)

    ax.set_xlabel('Age (Years, Log Scale)', fontsize=14, weight='bold')
    ax.set_ylabel('Z-Score', fontsize=14, weight='bold')
    ax.set_title(f'{gene_name} Expression Profile Across Age\nSequence: {sequence[:25]}{"..." if len(sequence) > 25 else ""}',
                 fontsize=16, weight='bold', pad=15)

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=11, frameon=True, edgecolor='black')

    props_text = (f"Length: {seq_properties['length']} aa\n"
                  f"Hydrophobicity: {seq_properties['avg_hydrophobicity']:.2f}\n"
                  f"Charge: {seq_properties['avg_charge']:.2f}\n"
                  f"Volume: {seq_properties['avg_volume']:.1f} Å³")
    ax.text(0.03, 0.97, props_text, transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()

    if save_plot:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"expression_plot_{gene_name}_{timestamp}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight', format='png')
        logger.info(f"Plot saved to {filename}")

    plt.show()

def main():
    logger.info("Starting expression plot generator...")
    print("Enter an amino acid sequence:")
    sequence = input().strip().upper()
    try:
        sequence = validate_amino_acid_sequence(sequence)
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        sys.exit(1)

    gene_name = input("Enter gene name: ").strip()
    create_expression_plot(sequence, gene_name=gene_name, save_plot=True)

if __name__ == "__main__":
    main()