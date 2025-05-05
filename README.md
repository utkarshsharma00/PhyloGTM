# PhyloGTM: Evaluating Guide Tree Merger with Alternative Tree Estimation Methods

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

PhyloGTM is a comprehensive evaluation framework for testing the Guide Tree Merger (GTM) method with various tree estimation approaches for single-gene phylogenetic analysis. This project compares the original GTM configuration (using FastTree for both guide and subset trees) against three hybrid approaches to identify combinations that provide an optimal balance between computational efficiency and phylogenetic accuracy.

### Key Features

- Implementation of multiple phylogenetic tree estimation methods:
  - FastTree (GTR and JC models)
  - Neighbor Joining (with JC, LogDet, and p-distance measures)
  - FastME with Balanced Minimum Evolution (BME)
  
- Four GTM configurations:
  - Original GTM: FastTree guide + FastTree subset trees
  - Hybrid GTM 1: FastTree guide + NJ-LogDet subset trees
  - Hybrid GTM 2: NJ-LogDet guide + FastTree subset trees
  - Hybrid GTM 3: NJ-LogDet guide + NJ-LogDet subset trees
  
- Comprehensive evaluation metrics:
  - Robinson-Foulds distance
  - False Negative (FN) rate
  - False Positive (FP) rate
  - Runtime performance
  - Memory usage

## Key Findings

Our analysis revealed several important insights:

1. **Guide tree quality matters**: FastTree guide trees consistently outperform NJ-LogDet guide trees for accuracy.

2. **Subset tree quality is even more critical**: High-quality FastTree subset trees can partially compensate for a lower-quality guide tree.

3. **Practical recommendations**:
   - For maximum accuracy: Use Original GTM (FastTree for both guide and subset trees)
   - For reasonable balance of speed and accuracy: Consider Hybrid GTM 2 (NJ-LogDet guide + FastTree subset trees)
   - For applications where speed is critical but some accuracy can be sacrificed: Use Hybrid GTM 3 or direct NJ methods

4. **Performance trade-offs**: Distance-based methods (FastME, NJ variants) are 7-20 times faster than ML methods but have substantially reduced accuracy.

## Table of Contents

- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Pipeline Implementation](#pipeline-implementation)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

```bash
# Clone the repository
git clone https://github.com/utkarshsharma00/PhyloGTM.git
cd PhyloGTM

# Create and activate conda environment (recommended)
conda create -n phylogtm python=3.9
conda activate phylogtm

# Install required packages
pip install -r requirements.txt

# Install external dependencies (FastTree and FastME)
# For Ubuntu/Debian:
sudo apt-get install fasttree fastme

# For macOS (using Homebrew):
brew install fasttree
brew install fastme
```

## Dependencies

### Python Packages
- dendropy==4.5.2
- biopython
- pandas
- numpy
- matplotlib
- seaborn
- plotly

### External Programs
- FastTree (version 2.1.11 or later)
- FastME (version 2.1.5 or later)

## Usage

### Basic Usage

```bash
# Run the main analysis script
./run_phylogeny_analysis.sh
```

### Custom Analysis

You can modify the parameters in the `run_phylogeny_analysis.sh` script to customize the analysis:

```bash
# Set custom dataset directory
DATASET_DIR="/path/to/your/datasets"

# Set specific model conditions to analyze
MODEL_COND="1000M1"

# Run only specific methods
# Modify the script to include/exclude methods as needed
```

### Example: Running Individual Methods

```bash
# Run FastTree (GTR model) on a dataset
FastTree -gtr -nt -fastest data/1000M1/R0/rose.aln.true.fasta > results/fasttree_gtr.tree

# Run NJ with LogDet distances
python scripts/convert_fasta_to_phylip.py data/1000M1/R0/rose.aln.true.fasta data/1000M1/R0/alignment.phy
fastme -i data/1000M1/R0/alignment.phy -o results/nj_logdet.tree -mN -dL

# Run Original GTM
# 1. Generate guide tree
FastTree -gtr -nt -fastest data/1000M1/R0/rose.aln.true.fasta > results/guide_tree.tree
# 2. Decompose dataset
python scripts/decompose.py --input-tree results/guide_tree.tree --sequence-file data/1000M1/R0/rose.aln.true.fasta --output-prefix results/subsets/subset --maximum-size 250 --mode "centroid"
# 3. Estimate subset trees
# (Loop through each subset file and run FastTree)
# 4. Merge subset trees
python scripts/gtm.py -s results/guide_tree.tree -t results/subsets/subset*.tree -o results/merged_tree.tree -m "convex"
```

## Project Structure

```
PhyloGTM/
├── data/                   # Dataset directory
│   ├── 1000M1/             # Model condition 1
│   │   ├── R0/             # Replicate 0
│   │   ├── R1/             # Replicate 1
│   │   └── ...
│   └── 1000M4/             # Model condition 4
│       ├── R0/
│       └── ...
├── scripts/                # Analysis scripts
│   ├── analyze_performance.py      # Performance analysis
│   ├── convert_fasta_to_phylip.py  # Format conversion
│   ├── convert_subset_to_phylip.py # Subset format conversion
│   ├── create_comparison_plots.py  # Visualization
│   ├── debug_guide_tree.py         # Guide tree debugging
│   ├── decompose.py                # Dataset decomposition
│   ├── decomposer.py               # Core decomposition logic
│   ├── fix_tree_format.py          # Tree format fixing
│   ├── gtm.py                      # Guide Tree Merger implementation
│   ├── gtm_old.py                  # Original GTM implementation
|   ├── run_phylogeny_analysis.sh      # Main execution script
│   ├── runtime_memory_plots.py     # Runtime visualization
│   ├── summarize_results.py        # Results summarization
│   ├── summarize_gtm_results.py    # GTM-specific results
│   ├── treecompare.py              # Tree comparison metrics
│   ├── treeutils.py                # Tree manipulation utilities
│   └── verify_guide_tree.py        # Guide tree verification
├── results/                # Analysis results
│   ├── baseline/           # Baseline method results
│   ├── original_gtm/       # Original GTM results
│   ├── hybrid_gtm1/        # Hybrid GTM 1 results
│   ├── hybrid_gtm2/        # Hybrid GTM 2 results
│   ├── hybrid_gtm3/        # Hybrid GTM 3 results
│   ├── performance/        # Performance metrics
│   └── summary/            # Summary statistics and plots
├── requirements.txt               # Python dependencies
└── README.md                      # This documentation
```

## Pipeline Implementation

The GTM pipeline consists of the following steps:

1. **Guide Tree Estimation**: Generate a guide tree using either FastTree (GTR model) or NJ-LogDet
2. **Dataset Decomposition**: Using the decompose.py script to decompose the dataset into subsets based on the guide tree
3. **Subset Tree Estimation**: Generate trees for each subset using either FastTree (GTR model) or NJ-LogDet
4. **Tree Merging**: Merge subset trees using the gtm.py script with the centroid merge mode
5. **Evaluation**: Compare the resulting trees to the true trees using Robinson-Foulds distance, FN rate, and FP rate

### Command Details

#### 1. Data Preparation
```bash
# Convert FASTA to PHYLIP format for distance-based methods
python scripts/convert_fasta_to_phylip.py input_alignment.fasta output_alignment.phy
```

#### 2. Baseline Methods
```bash
# FastTree (GTR model)
FastTree -gtr -nt -fastest input_alignment.fasta > fasttree_gtr.tree

# NJ with LogDet distances
fastme -i alignment.phy -o nj_logdet.tree -mN -dL
```

#### 3. GTM Pipeline
```bash
# Guide Tree Estimation
FastTree -gtr -nt -fastest input_alignment.fasta > guide_tree.tree
# or
fastme -i alignment.phy -o guide_tree.tree -mN -dL

# Dataset Decomposition
python scripts/decompose.py --input-tree guide_tree.tree --sequence-file input_alignment.fasta --output-prefix subsets/subset --maximum-size 250 --mode "centroid"

# Subset Tree Estimation (for FastTree subsets)
# Loop through each subset
for SUBSET_FILE in subsets/subset*.out; do
    FastTree -gtr -nt -fastest $SUBSET_FILE > ${SUBSET_FILE%.out}.tree
done

# Tree Merging
python scripts/gtm.py -s guide_tree.tree -t subsets/subset*.tree -o merged_tree.tree -m "convex"
```

## Evaluation Metrics

We use the following metrics to evaluate the performance of each method:

- **Robinson-Foulds (RF) Distance**: The total number of bipartitions that differ between the estimated tree and the reference tree.
- **False Negative (FN) Rate**: The proportion of bipartitions in the reference tree that are missing from the estimated tree.
- **False Positive (FP) Rate**: The proportion of bipartitions in the estimated tree that are not in the reference tree.
- **Runtime**: The wall-clock time required to complete the entire tree estimation process.
- **Memory Usage**: The peak memory usage during tree estimation.

These metrics are calculated using the `treecompare.py` script:

```bash
python scripts/treecompare.py estimated_tree.tree reference_tree.tree comparison.txt
```

## Results

Our experimental results comparing different GTM configurations on the 1000M1 and 1000M4 datasets showed:

### Tree Accuracy Metrics

| Method | 1000M1 |  |  | 1000M4 |  |  |
|--------|----------|---------|---------|----------|---------|---------|
|  | RF (norm) | FN | FP | RF (norm) | FN | FP |
| Original GTM | 0.446 | 0.109 | 0.112 | 0.258 | 0.053 | 0.076 |
| Hybrid GTM 1 | 0.934 | 0.230 | 0.232 | 0.520 | 0.119 | 0.140 |
| Hybrid GTM 2 | 0.500 | 0.123 | 0.125 | 0.270 | 0.056 | 0.079 |
| Hybrid GTM 3 | 0.980 | 0.241 | 0.244 | 0.530 | 0.122 | 0.143 |

### Runtime Performance

| Method | Avg Runtime | Speedup vs Original GTM | Avg Memory (GB) |
|--------|-------------|------------------------|-----------------|
| FastME (BME) | 4m 1s | 20.6× faster | 1.88 |
| NJ (JC) | 4m 22s | 19.0× faster | 1.87 |
| NJ (p-distance) | 10m 7s | 8.2× faster | 1.88 |
| NJ (LogDet) | 11m 28s | 7.2× faster | 1.88 |
| FastTree (JC) | 11m 40s | 7.1× faster | 1.88 |
| FastTree (GTR) | 14m 31s | 5.7× faster | 1.88 |
| Hybrid GTM 3 | 17m 21s | 4.8× faster | 1.90 |
| Hybrid GTM 1 | 20m 29s | 4.1× faster | 1.91 |
| Hybrid GTM 2 | 69m 27s | 1.2× faster | 1.91 |
| Original GTM | 83m 4s | 1.0× | 1.91 |

For detailed visualizations and further analysis, please refer to the plots generated in the `results/summary/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
Sharma, U. (2025). Evaluating Guide Tree Merger with Alternative Tree Estimation Methods for Single-Gene Phylogenetic Analysis. University of Illinois Urbana-Champaign.
```

## Acknowledgements

- This project was developed as part of a course at the University of Illinois Urbana-Champaign.
- GTM implementation is based on the work by [Park et al., 2021](https://www.mdpi.com/1999-4893/14/5/148).
- Special thanks to all collaborators and contributors to the phylogenetic methods used in this project.
