#!/bin/bash
# run_phylogeny_analysis.sh
# 
# Main SLURM job script for running all phylogenetic analyses

# #SBATCH --time=12:00:00                     # Job run time (hh:mm:ss)
# #SBATCH --nodes=1                           # Number of nodes
# #SBATCH --ntasks-per-node=32                # 32 cores
# #SBATCH --mem=32G                           # Memory allocation
# #SBATCH --job-name=phylogeny-analysis       # Name of job
# #SBATCH --account=25sp-cs581a-eng           # Account for CS 581 students
# #SBATCH --partition=eng-instruction         # Partition for CS 581
# #SBATCH --constraint=AE7713                 # Target the 128-core nodes
# #SBATCH --output=phylogeny_analysis_%j.log  # Output file with job ID
# #SBATCH --mail-type=BEGIN,END,FAIL          # Mail events
# #SBATCH --mail-user=usharma4@illinois.edu   # Your email

# Directory setup
# PROJ_DIR="/u/usharma4/phylogeny_project"
PROJ_DIR="/Users/utkarsh/Spring_2025/phylogeny_project"
DATASET_DIR="$PROJ_DIR/data"
SCRIPTS_DIR="$PROJ_DIR/scripts"
RESULTS_DIR="$PROJ_DIR/results"
FASTME_FILEPATH="/usr/local/bin/fastme"  

# Create results directories
mkdir -p $RESULTS_DIR/baseline
mkdir -p $RESULTS_DIR/original_gtm
mkdir -p $RESULTS_DIR/hybrid_gtm1
mkdir -p $RESULTS_DIR/hybrid_gtm2
mkdir -p $RESULTS_DIR/hybrid_gtm3
mkdir -p $RESULTS_DIR/summary
mkdir -p $RESULTS_DIR/performance

# Function to measure time in a portable way
measure_performance() {
    local method=$1
    local command=$2
    local model=$3
    local rep=$4
    local output_file="$RESULTS_DIR/performance/${method}_${model}_R${rep}.perf"
    
    echo "    Measuring performance for ${method}..."
    
    # Start time with higher precision
    start_time=$(date +%s.%N)  # Use nanoseconds for higher precision
    
    # Execute the command
    eval $command
    cmd_status=$?
    
    # End time with higher precision
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc -l)  # Use bc for floating point arithmetic
    
    # Get memory info if possible (using ps)
    mem_kb="NA"
    if command -v ps &> /dev/null; then
        mem_kb=$(ps -o rss= -p $$ | tail -1)
        if [ -z "$mem_kb" ]; then
            mem_kb="NA"
        fi
    fi
    
    # Save performance data with 6 decimal places for elapsed time
    printf "%.6f,%s,NA\n" $elapsed $mem_kb > "$output_file"
    
    # Return the original command's exit status
    return $cmd_status
}

# Define the path to Python in the conda environment
PYTHON_PATH=$(which python)
echo "Using Python from: $PYTHON_PATH"

# Verify packages are installed
echo "Verifying installed packages:"
$PYTHON_PATH -c "
import sys
print(f'Python path: {sys.path}')
try:
    import dendropy
    print(f'DendroPy version: {dendropy.__version__}')
except ImportError as e:
    print(f'ERROR: DendroPy not installed: {e}')
    sys.exit(1)

try:
    from Bio import AlignIO
    print('Biopython is installed')
except ImportError as e:
    print(f'ERROR: Biopython not installed: {e}')
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print('pandas, numpy, and matplotlib are installed')
except ImportError as e:
    print(f'ERROR: pandas, numpy, or matplotlib not installed: {e}')
    sys.exit(1)
"

# Check exit status of verification
if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed. Exiting."
    exit 1
fi

# Make sure scripts directory is in Python path
export PYTHONPATH="$SCRIPTS_DIR:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Verify data directory structure
echo "Verifying data directory structure:"
ls -la $DATASET_DIR
if [ -d "$DATASET_DIR/1000M1" ]; then
    ls -la $DATASET_DIR/1000M1
    if [ -d "$DATASET_DIR/1000M1/R0" ]; then
        ls -la $DATASET_DIR/1000M1/R0
    fi
    if [ -d "$DATASET_DIR/1000M1/R1" ]; then
        ls -la $DATASET_DIR/1000M1/R1
    fi
else
    echo "ERROR: Dataset directory structure is incorrect!"
    echo "Expected: $DATASET_DIR/1000M1/R0, $DATASET_DIR/1000M1/R1, etc."
    exit 1
fi

# Process datasets
for MODEL_COND in "1000M1" "1000M4"; do
    echo "Processing model condition: $MODEL_COND"
    
    # Skip if model condition directory doesn't exist
    if [ ! -d "$DATASET_DIR/$MODEL_COND" ]; then
        echo "  Model condition directory $DATASET_DIR/$MODEL_COND not found, skipping"
        continue
    fi
    
    mkdir -p $RESULTS_DIR/baseline/$MODEL_COND
    mkdir -p $RESULTS_DIR/original_gtm/$MODEL_COND
    mkdir -p $RESULTS_DIR/hybrid_gtm1/$MODEL_COND
    mkdir -p $RESULTS_DIR/hybrid_gtm2/$MODEL_COND
    mkdir -p $RESULTS_DIR/hybrid_gtm3/$MODEL_COND
    
    # Process replicates (R0 through R19)
    for REP in {0..19}; do
        REP_DIR="$DATASET_DIR/$MODEL_COND/R$REP"
        
        # Skip if the replicate directory doesn't exist
        if [ ! -d "$REP_DIR" ]; then
            echo "  Replicate directory $REP_DIR not found, skipping"
            continue
        fi
        
        echo "  Processing replicate R$REP"
        
        # Define input files
        if [ -f "$REP_DIR/rose.aln.true.fasta" ]; then
            ALIGNMENT="$REP_DIR/rose.aln.true.fasta"
        else
            echo "WARNING: Cannot find alignment file in $REP_DIR"
            echo "Files in $REP_DIR:"
            ls -la $REP_DIR
            continue
        fi
        
        if [ -f "$REP_DIR/rose.tt" ]; then
            TRUE_TREE="$REP_DIR/rose.tt"
        elif [ -f "$REP_DIR/random.tree" ]; then
            TRUE_TREE="$REP_DIR/random.tree"
        else
            echo "WARNING: Cannot find true tree file in $REP_DIR"
            echo "Files in $REP_DIR:"
            ls -la $REP_DIR
            continue
        fi
        
        # Define output directories
        BASELINE_DIR="$RESULTS_DIR/baseline/$MODEL_COND/R$REP"
        ORIGINAL_GTM_DIR="$RESULTS_DIR/original_gtm/$MODEL_COND/R$REP"
        HYBRID_GTM1_DIR="$RESULTS_DIR/hybrid_gtm1/$MODEL_COND/R$REP"
        HYBRID_GTM2_DIR="$RESULTS_DIR/hybrid_gtm2/$MODEL_COND/R$REP"
        HYBRID_GTM3_DIR="$RESULTS_DIR/hybrid_gtm3/$MODEL_COND/R$REP"
        
        mkdir -p $BASELINE_DIR
        mkdir -p $ORIGINAL_GTM_DIR
        mkdir -p $HYBRID_GTM1_DIR
        mkdir -p $HYBRID_GTM2_DIR
        mkdir -p $HYBRID_GTM3_DIR
        
        # Convert FASTA to PHYLIP format for FastME
        PHYLIP_FILE="$BASELINE_DIR/alignment.phy"
        echo "  Converting FASTA to PHYLIP format for FastME"
        measure_performance "convert_phylip" "$PYTHON_PATH $SCRIPTS_DIR/convert_fasta_to_phylip.py \"$ALIGNMENT\" \"$PHYLIP_FILE\"" "$MODEL_COND" "$REP"
        
        # Continue only if PHYLIP file was created successfully
        if [ ! -f "$PHYLIP_FILE" ]; then
            echo "  ERROR: PHYLIP file $PHYLIP_FILE was not created. Skipping this replicate."
            continue
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 1: FastTree with GTR model
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/fasttree_gtr.tree" ]; then
            echo "    Running FastTree GTR (Baseline)"
            measure_performance "fasttree_gtr" "FastTree -gtr -nt -fastest \"$ALIGNMENT\" > \"$BASELINE_DIR/fasttree_gtr.tree\"" "$MODEL_COND" "$REP"
            
            # Check if tree was created successfully
            if [ ! -s "$BASELINE_DIR/fasttree_gtr.tree" ]; then
                echo "    ERROR: FastTree GTR failed to create a valid tree"
            else
                echo "    FastTree GTR completed successfully"
            fi
        else
            echo "    FastTree GTR already completed, skipping"
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 2: FastTree with JC model
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/fasttree_jc.tree" ]; then
            echo "    Running FastTree JC (Baseline)"
            measure_performance "fasttree_jc" "FastTree -nt -fastest \"$ALIGNMENT\" > \"$BASELINE_DIR/fasttree_jc.tree\"" "$MODEL_COND" "$REP"
            
            # Check if tree was created successfully
            if [ ! -s "$BASELINE_DIR/fasttree_jc.tree" ]; then
                echo "    ERROR: FastTree JC failed to create a valid tree"
            else
                echo "    FastTree JC completed successfully"
            fi
        else
            echo "    FastTree JC already completed, skipping"
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 3: NJ with JC distances
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/nj_jc.tree" ] || [ ! -s "$BASELINE_DIR/nj_jc.tree" ]; then
            echo "    Running FastME with JC distances (Baseline)"
            if [ -f "$PHYLIP_FILE" ]; then
                measure_performance "nj_jc" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$BASELINE_DIR/nj_jc.tree\" -mN -dJ" "$MODEL_COND" "$REP"
                
                # Check if tree was created successfully
                if [ ! -s "$BASELINE_DIR/nj_jc.tree" ]; then
                    echo "    ERROR: FastME with JC distances failed to create a valid tree"
                else
                    echo "    FastME with JC distances completed successfully"
                fi
            else
                echo "    ERROR: PHYLIP file not found for FastME with JC distances"
            fi
        else
            echo "    NJ with JC distances already completed, skipping"
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 4: NJ with LogDet distances
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/nj_logdet.tree" ] || [ ! -s "$BASELINE_DIR/nj_logdet.tree" ]; then
            echo "    Running FastME with LogDet distances (Baseline)"
            if [ -f "$PHYLIP_FILE" ]; then
                measure_performance "nj_logdet" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$BASELINE_DIR/nj_logdet.tree\" -mN -dL" "$MODEL_COND" "$REP"
                
                # Check if tree was created successfully
                if [ ! -s "$BASELINE_DIR/nj_logdet.tree" ]; then
                    echo "    ERROR: FastME with LogDet distances failed to create a valid tree"
                else
                    echo "    FastME with LogDet distances completed successfully"
                fi
            else
                echo "    ERROR: PHYLIP file not found for FastME with LogDet distances"
            fi
        else
            echo "    NJ with LogDet distances already completed, skipping"
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 5: NJ with p-distances
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/nj_pdist.tree" ] || [ ! -s "$BASELINE_DIR/nj_pdist.tree" ]; then
            echo "    Running FastME with p-distances (Baseline)"
            if [ -f "$PHYLIP_FILE" ]; then
                measure_performance "nj_pdist" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$BASELINE_DIR/nj_pdist.tree\" -mN -dp" "$MODEL_COND" "$REP"
                
                # Check if tree was created successfully
                if [ ! -s "$BASELINE_DIR/nj_pdist.tree" ]; then
                    echo "    ERROR: FastME with p-distances failed to create a valid tree"
                else
                    echo "    FastME with p-distances completed successfully"
                fi
            else
                echo "    ERROR: PHYLIP file not found for FastME with p-distances"
            fi
        else
            echo "    NJ with p-distances already completed, skipping"
        fi
        
        #----------------------------------------------------
        # BASELINE METHOD 7: FastME with BME
        #----------------------------------------------------
        if [ ! -f "$BASELINE_DIR/fastme_bme.tree" ] || [ ! -s "$BASELINE_DIR/fastme_bme.tree" ]; then
            echo "    Running FastME with BME criterion (Baseline)"
            if [ -f "$PHYLIP_FILE" ]; then
                measure_performance "fastme_bme" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$BASELINE_DIR/fastme_bme.tree\" -mB -dJ" "$MODEL_COND" "$REP"
                
                # Check if tree was created successfully
                if [ ! -s "$BASELINE_DIR/fastme_bme.tree" ]; then
                    echo "    ERROR: FastME with BME criterion failed to create a valid tree"
                else
                    echo "    FastME with BME criterion completed successfully"
                fi
            else
                echo "    ERROR: PHYLIP file not found for FastME with BME criterion"
            fi
        else
            echo "    FastME with BME already completed, skipping"
        fi
        
        #----------------------------------------------------
        # Fix tree files before comparison
        #----------------------------------------------------
        for tree_file in "$BASELINE_DIR"/*.tree; do
            if [ -f "$tree_file" ] && [ -s "$tree_file" ]; then
                # Fix potential format issues with dendropy
                measure_performance "fix_tree_format" "$PYTHON_PATH $SCRIPTS_DIR/fix_tree_format.py \"$tree_file\"" "$MODEL_COND" "$REP"
                
                # Replace original with fixed version if successful
                if [ -f "$tree_file.fixed" ]; then
                    mv "$tree_file.fixed" "$tree_file"
                fi
            fi
        done
        
        #----------------------------------------------------
        # ORIGINAL GTM: FastTree guide + FastTree subset trees
        #----------------------------------------------------
        # 1. Generate FastTree guide tree if not already done
        if [ ! -f "$ORIGINAL_GTM_DIR/guide_tree.tree" ] || [ ! -s "$ORIGINAL_GTM_DIR/guide_tree.tree" ]; then
            echo "    Generating FastTree guide tree for Original GTM"
            mkdir -p "$ORIGINAL_GTM_DIR"
            
            # Check if FastTree is available
            if ! command -v FastTree &> /dev/null; then
                echo "    ERROR: FastTree command not found. Please install FastTree."
                continue
            fi
            
            measure_performance "original_gtm_guide" "FastTree -gtr -nt -fastest \"$ALIGNMENT\" > \"$ORIGINAL_GTM_DIR/guide_tree.tree.tmp\"" "$MODEL_COND" "$REP"
            
            echo "Examining tree file contents:"
            if [ -f "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp" ]; then
                file_size=$(wc -c < "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp")
                echo "File size: $file_size bytes"
                head -n 1 "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp" | cut -c1-100
                echo ""
            else
                echo "File does not exist"
            fi
            
            # Check if guide tree was created successfully
            if [ ! -s "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp" ]; then
                echo "    ERROR: FastTree failed to create a valid guide tree for Original GTM"
                continue
            else
                # Verify it's a valid Newick file - fixed indentation
                $PYTHON_PATH $SCRIPTS_DIR/verify_guide_tree.py "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp"
                if [ $? -eq 0 ]; then
                    mv "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp" "$ORIGINAL_GTM_DIR/guide_tree.tree"
                    echo "    FastTree guide tree for Original GTM created successfully"
                else
                    echo "    ERROR: FastTree produced an invalid guide tree"
                    head -n 5 "$ORIGINAL_GTM_DIR/guide_tree.tree.tmp"
                    continue
                fi
            fi
        fi
        
        # Debug guide tree file
        echo "    Checking guide tree file:"
        if [ -f "$ORIGINAL_GTM_DIR/guide_tree.tree" ]; then
            ls -la "$ORIGINAL_GTM_DIR/guide_tree.tree"
            head -c 100 "$ORIGINAL_GTM_DIR/guide_tree.tree"
            echo ""
        else
            echo "    Guide tree file not found!"
        fi

        # 2. Decompose dataset using FastTree guide tree
        if [ ! -d "$ORIGINAL_GTM_DIR/subsets" ] || [ $(ls -1 "$ORIGINAL_GTM_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
            echo "    Decomposing dataset using FastTree guide tree"
            mkdir -p "$ORIGINAL_GTM_DIR/subsets"
            
            # Run decomposition with verbose error handling
            measure_performance "original_gtm_decompose" "$PYTHON_PATH $SCRIPTS_DIR/decompose.py --input-tree \"$ORIGINAL_GTM_DIR/guide_tree.tree\" --sequence-file \"$ALIGNMENT\" --output-prefix \"$ORIGINAL_GTM_DIR/subsets/subset\" --maximum-size 250 --mode \"centroid\" 2>&1 | tee \"$ORIGINAL_GTM_DIR/decompose_log.txt\"" "$MODEL_COND" "$REP"
            
            # Check decomposition result
            if [ ${PIPESTATUS[0]} -ne 0 ]; then
                echo "    ERROR: decompose.py failed. See log at $ORIGINAL_GTM_DIR/decompose_log.txt"
                echo "    Trying to debug the issue:"
                
                # Debug information - fixed indentation
                $PYTHON_PATH $SCRIPTS_DIR/debug_guide_tree.py "$ORIGINAL_GTM_DIR/guide_tree.tree" "$ALIGNMENT"
                continue
            fi
            
            # Check if decomposition was successful
            if [ $(ls -1 "$ORIGINAL_GTM_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
                echo "    ERROR: Failed to decompose dataset for Original GTM"
                continue
            else
                echo "    Successfully decomposed dataset for Original GTM"
            fi
        fi
        
        # 3. Run FastTree on each subset
        SUBSET_COUNT=$(ls -1 "$ORIGINAL_GTM_DIR/subsets/subset"*.out 2>/dev/null | wc -l)
        if [ "$SUBSET_COUNT" -gt 0 ]; then
            echo "    Running FastTree on subsets for Original GTM"
            for SUBSET_FILE in "$ORIGINAL_GTM_DIR/subsets/subset"*.out; do
                SUBSET_PREFIX="${SUBSET_FILE%.out}"
                SUBSET_TREE="$SUBSET_PREFIX.tree"
                SUBSET_NUM=$(basename "$SUBSET_PREFIX" | sed 's/subset//')
                
                if [ ! -f "$SUBSET_TREE" ] || [ ! -s "$SUBSET_TREE" ]; then
                    echo "      Processing subset $(basename "$SUBSET_FILE")"
                    measure_performance "original_gtm_subset_tree_$SUBSET_NUM" "FastTree -gtr -nt -fastest \"$SUBSET_FILE\" > \"$SUBSET_TREE\"" "$MODEL_COND" "$REP"
                    
                    # Check if subset tree was created successfully
                    if [ ! -s "$SUBSET_TREE" ]; then
                        echo "      ERROR: FastTree failed to create a valid tree for subset $(basename "$SUBSET_FILE")"
                    else
                        echo "      Successfully created tree for subset $(basename "$SUBSET_FILE")"
                    fi
                fi
            done
            
            # Check if all subset trees were created
            TREE_COUNT=$(ls -1 "$ORIGINAL_GTM_DIR/subsets/subset"*.tree 2>/dev/null | wc -l)
            if [ "$TREE_COUNT" -ne "$SUBSET_COUNT" ]; then
                echo "    ERROR: Not all subset trees were created for Original GTM"
                echo "    Found $TREE_COUNT trees for $SUBSET_COUNT subsets"
            fi
        fi
        
        # 4. Merge subset trees using GTM
        if [ "$SUBSET_COUNT" -gt 0 ] && [ ! -f "$ORIGINAL_GTM_DIR/merged_tree.tree" ]; then
            echo "    Merging subset trees using GTM (Original GTM)"
            
            # Create list of subset trees
            SUBSET_TREES=()
            for SUBSET_TREE in "$ORIGINAL_GTM_DIR/subsets/subset"*.tree; do
                if [ -f "$SUBSET_TREE" ] && [ -s "$SUBSET_TREE" ]; then
                    SUBSET_TREES+=("$SUBSET_TREE")
                fi
            done
            
            # Check if we have subset trees to merge
            if [ ${#SUBSET_TREES[@]} -gt 0 ]; then
                # Run GTM - create a string with all subset trees
                SUBSET_TREE_STR="${SUBSET_TREES[@]}"
                measure_performance "original_gtm_merge" "$PYTHON_PATH $SCRIPTS_DIR/gtm.py -s \"$ORIGINAL_GTM_DIR/guide_tree.tree\" -t $SUBSET_TREE_STR -o \"$ORIGINAL_GTM_DIR/merged_tree.tree\" -m \"convex\"" "$MODEL_COND" "$REP"
                
                # Check if merged tree was created successfully
                if [ ! -s "$ORIGINAL_GTM_DIR/merged_tree.tree" ]; then
                    echo "    ERROR: GTM failed to create a valid merged tree for Original GTM"
                else
                    echo "    Successfully created merged tree for Original GTM"
                fi
            else
                echo "    ERROR: No valid subset trees found for merging in Original GTM"
            fi
        fi
        
        #----------------------------------------------------
        # HYBRID GTM 1: FastTree guide + NJ-LogDet subset trees
        #----------------------------------------------------
        # 1. Reuse the FastTree guide tree from Original GTM
        if [ ! -f "$HYBRID_GTM1_DIR/guide_tree.tree" ]; then
            if [ -f "$ORIGINAL_GTM_DIR/guide_tree.tree" ] && [ -s "$ORIGINAL_GTM_DIR/guide_tree.tree" ]; then
                echo "    Copying FastTree guide tree for Hybrid GTM 1"
                cp "$ORIGINAL_GTM_DIR/guide_tree.tree" "$HYBRID_GTM1_DIR/guide_tree.tree"
                # Record dependency
                mkdir -p "$HYBRID_GTM1_DIR"
                echo "fasttree_gtr" > "$HYBRID_GTM1_DIR/dependencies.txt"
            else
                echo "    Generating FastTree guide tree for Hybrid GTM 1"
                measure_performance "hybrid_gtm1_guide" "FastTree -gtr -nt -fastest \"$ALIGNMENT\" > \"$HYBRID_GTM1_DIR/guide_tree.tree.tmp\"" "$MODEL_COND" "$REP"
                
                # Verify it's a valid Newick file - fixed indentation
                $PYTHON_PATH $SCRIPTS_DIR/verify_guide_tree.py "$HYBRID_GTM1_DIR/guide_tree.tree.tmp"
                if [ $? -eq 0 ]; then
                    mv "$HYBRID_GTM1_DIR/guide_tree.tree.tmp" "$HYBRID_GTM1_DIR/guide_tree.tree"
                else
                    echo "    ERROR: FastTree failed to create a valid guide tree for Hybrid GTM 1"
                    continue
                fi
            fi
        fi
        
        # 2. Decompose dataset using FastTree guide tree
        if [ ! -d "$HYBRID_GTM1_DIR/subsets" ] || [ $(ls -1 "$HYBRID_GTM1_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
            echo "    Decomposing dataset using FastTree guide tree for Hybrid GTM 1"
            mkdir -p "$HYBRID_GTM1_DIR/subsets"
            measure_performance "hybrid_gtm1_decompose" "$PYTHON_PATH $SCRIPTS_DIR/decompose.py --input-tree \"$HYBRID_GTM1_DIR/guide_tree.tree\" --sequence-file \"$ALIGNMENT\" --output-prefix \"$HYBRID_GTM1_DIR/subsets/subset\" --maximum-size 250 --mode \"centroid\"" "$MODEL_COND" "$REP"
            
            # Check if decomposition was successful
            if [ $(ls -1 "$HYBRID_GTM1_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
                echo "    ERROR: Failed to decompose dataset for Hybrid GTM 1"
                continue
            fi
        fi
        
        # 3. Run NJ-LogDet on each subset
        SUBSET_COUNT=$(ls -1 "$HYBRID_GTM1_DIR/subsets/subset"*.out 2>/dev/null | wc -l)
        echo "    Found $SUBSET_COUNT subsets for Hybrid GTM 1"
        if [ "$SUBSET_COUNT" -gt 0 ]; then
            echo "    Running NJ-LogDet on subsets for Hybrid GTM 1"
            for SUBSET_FILE in "$HYBRID_GTM1_DIR/subsets/subset"*.out; do
                SUBSET_PREFIX="${SUBSET_FILE%.out}"
                SUBSET_TREE="$SUBSET_PREFIX.tree"
                SUBSET_PHY="$SUBSET_PREFIX.phy"
                SUBSET_NUM=$(basename "$SUBSET_PREFIX" | sed 's/subset//')
                
                if [ ! -f "$SUBSET_TREE" ] || [ ! -s "$SUBSET_TREE" ]; then
                    echo "      Processing subset $(basename "$SUBSET_FILE")"
                    
                    # Convert to PHYLIP
                    measure_performance "hybrid_gtm1_convert_phy_$SUBSET_NUM" "$PYTHON_PATH $SCRIPTS_DIR/convert_subset_to_phylip.py \"$SUBSET_FILE\" \"$SUBSET_PHY\"" "$MODEL_COND" "$REP"

                    # Check if PHYLIP conversion was successful
                    if [ ! -f "$SUBSET_PHY" ] || [ ! -s "$SUBSET_PHY" ]; then
                        echo "      ERROR: Failed to convert subset to PHYLIP format"
                        continue
                    fi
                    
                    # Run FastME with NJ and LogDet
                    measure_performance "hybrid_gtm1_subset_tree_$SUBSET_NUM" "\"$FASTME_FILEPATH\" -i \"$SUBSET_PHY\" -o \"$SUBSET_TREE\" -mN -dL" "$MODEL_COND" "$REP"
                    
                    # Check if tree was created successfully
                    if [ ! -s "$SUBSET_TREE" ]; then
                        echo "      ERROR: FastME failed to create a valid tree for subset"
                    else
                        echo "      Successfully created tree for subset"
                    fi
                fi
            done
            
            # Check if all subset trees were created
            TREE_COUNT=$(ls -1 "$HYBRID_GTM1_DIR/subsets/subset"*.tree 2>/dev/null | wc -l)
            if [ "$TREE_COUNT" -ne "$SUBSET_COUNT" ]; then
                echo "    ERROR: Not all subset trees were created for Hybrid GTM 1"
                echo "    Found $TREE_COUNT trees for $SUBSET_COUNT subsets"
            fi
        fi
        
        # 4. Merge subset trees using GTM
        if [ "$SUBSET_COUNT" -gt 0 ] && [ ! -f "$HYBRID_GTM1_DIR/merged_tree.tree" ]; then
            echo "    Merging subset trees using GTM (Hybrid GTM 1)"
            
            # Create list of subset trees
            SUBSET_TREES=()
            for SUBSET_TREE in "$HYBRID_GTM1_DIR/subsets/subset"*.tree; do
                if [ -f "$SUBSET_TREE" ] && [ -s "$SUBSET_TREE" ]; then
                    SUBSET_TREES+=("$SUBSET_TREE")
                fi
            done
            
            # Check if we have subset trees to merge
            if [ ${#SUBSET_TREES[@]} -gt 0 ]; then
                # Run GTM - create a string with all subset trees
                SUBSET_TREE_STR="${SUBSET_TREES[@]}"
                measure_performance "hybrid_gtm1_merge" "$PYTHON_PATH $SCRIPTS_DIR/gtm.py -s \"$HYBRID_GTM1_DIR/guide_tree.tree\" -t $SUBSET_TREE_STR -o \"$HYBRID_GTM1_DIR/merged_tree.tree\" -m \"convex\"" "$MODEL_COND" "$REP"
                
                # Check if merged tree was created successfully
                if [ ! -s "$HYBRID_GTM1_DIR/merged_tree.tree" ]; then
                    echo "    ERROR: GTM failed to create a valid merged tree for Hybrid GTM 1"
                else
                    echo "    Successfully created merged tree for Hybrid GTM 1"
                fi
            else
                echo "    ERROR: No valid subset trees found for merging in Hybrid GTM 1"
            fi
        fi
        
        #----------------------------------------------------
        # HYBRID GTM 2: NJ-LogDet guide + FastTree subset trees
        #----------------------------------------------------
        # 1. Generate NJ-LogDet guide tree if not already done
        if [ ! -f "$HYBRID_GTM2_DIR/guide_tree.tree" ]; then
            echo "    Generating NJ-LogDet guide tree for Hybrid GTM 2"
            if [ -f "$PHYLIP_FILE" ] && [ -s "$PHYLIP_FILE" ]; then
                measure_performance "hybrid_gtm2_guide" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$HYBRID_GTM2_DIR/guide_tree.tree\" -mN -dL" "$MODEL_COND" "$REP"
                # Record dependency
                mkdir -p "$HYBRID_GTM2_DIR"
                echo "nj_logdet" > "$HYBRID_GTM2_DIR/dependencies.txt"
                
                # Check if guide tree was created successfully
                if [ ! -s "$HYBRID_GTM2_DIR/guide_tree.tree" ]; then
                    echo "    ERROR: FastME failed to create a valid guide tree for Hybrid GTM 2"
                    continue
                else
                    echo "    Successfully created NJ-LogDet guide tree for Hybrid GTM 2"
                fi
            else
                echo "    ERROR: PHYLIP file not found for creating NJ-LogDet guide tree"
                continue
            fi
        fi
        
        # 2. Decompose dataset using NJ-LogDet guide tree
        if [ ! -d "$HYBRID_GTM2_DIR/subsets" ] || [ $(ls -1 "$HYBRID_GTM2_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
            echo "    Decomposing dataset using NJ-LogDet guide tree for Hybrid GTM 2"
            mkdir -p "$HYBRID_GTM2_DIR/subsets"
            measure_performance "hybrid_gtm2_decompose" "$PYTHON_PATH $SCRIPTS_DIR/decompose.py --input-tree \"$HYBRID_GTM2_DIR/guide_tree.tree\" --sequence-file \"$ALIGNMENT\" --output-prefix \"$HYBRID_GTM2_DIR/subsets/subset\" --maximum-size 250 --mode \"centroid\"" "$MODEL_COND" "$REP"
            
            # Check if decomposition was successful
            if [ $(ls -1 "$HYBRID_GTM2_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
                echo "    ERROR: Failed to decompose dataset for Hybrid GTM 2"
                continue
            else
                echo "    Successfully decomposed dataset for Hybrid GTM 2"
            fi
        fi
        
        # 3. Run FastTree on each subset
        SUBSET_COUNT=$(ls -1 "$HYBRID_GTM2_DIR/subsets/subset"*.out 2>/dev/null | wc -l)
        if [ "$SUBSET_COUNT" -gt 0 ]; then
            echo "    Running FastTree on subsets for Hybrid GTM 2"
            for SUBSET_FILE in "$HYBRID_GTM2_DIR/subsets/subset"*.out; do
                SUBSET_PREFIX="${SUBSET_FILE%.out}"
                SUBSET_TREE="$SUBSET_PREFIX.tree"
                SUBSET_NUM=$(basename "$SUBSET_PREFIX" | sed 's/subset//')
                
                if [ ! -f "$SUBSET_TREE" ] || [ ! -s "$SUBSET_TREE" ]; then
                    echo "      Processing subset $(basename "$SUBSET_FILE")"
                    measure_performance "hybrid_gtm2_subset_tree_$SUBSET_NUM" "FastTree -gtr -nt -fastest \"$SUBSET_FILE\" > \"$SUBSET_TREE\"" "$MODEL_COND" "$REP"
                    
                    # Check if subset tree was created successfully
                    if [ ! -s "$SUBSET_TREE" ]; then
                        echo "      ERROR: FastTree failed to create a valid tree for subset $(basename "$SUBSET_FILE")"
                    else
                        echo "      Successfully created tree for subset $(basename "$SUBSET_FILE")"
                    fi
                fi
            done
            
            # Check if all subset trees were created
            TREE_COUNT=$(ls -1 "$HYBRID_GTM2_DIR/subsets/subset"*.tree 2>/dev/null | wc -l)
            if [ "$TREE_COUNT" -ne "$SUBSET_COUNT" ]; then
                echo "    ERROR: Not all subset trees were created for Hybrid GTM 2"
                echo "    Found $TREE_COUNT trees for $SUBSET_COUNT subsets"
            fi
        fi
        
        # 4. Merge subset trees using GTM
        if [ "$SUBSET_COUNT" -gt 0 ] && [ ! -f "$HYBRID_GTM2_DIR/merged_tree.tree" ]; then
            echo "    Merging subset trees using GTM (Hybrid GTM 2)"
            
            # Create list of subset trees
            SUBSET_TREES=()
            for SUBSET_TREE in "$HYBRID_GTM2_DIR/subsets/subset"*.tree; do
                if [ -f "$SUBSET_TREE" ] && [ -s "$SUBSET_TREE" ]; then
                    SUBSET_TREES+=("$SUBSET_TREE")
                fi
            done
            
            # Check if we have subset trees to merge
            if [ ${#SUBSET_TREES[@]} -gt 0 ]; then
                # Run GTM - create a string with all subset trees
                SUBSET_TREE_STR="${SUBSET_TREES[@]}"
                measure_performance "hybrid_gtm2_merge" "$PYTHON_PATH $SCRIPTS_DIR/gtm.py -s \"$HYBRID_GTM2_DIR/guide_tree.tree\" -t $SUBSET_TREE_STR -o \"$HYBRID_GTM2_DIR/merged_tree.tree\" -m \"convex\"" "$MODEL_COND" "$REP"
                
                # Check if merged tree was created successfully
                if [ ! -s "$HYBRID_GTM2_DIR/merged_tree.tree" ]; then
                    echo "    ERROR: GTM failed to create a valid merged tree for Hybrid GTM 2"
                else
                    echo "    Successfully created merged tree for Hybrid GTM 2"
                fi
            else
                echo "    ERROR: No valid subset trees found for merging in Hybrid GTM 2"
            fi
        fi
        
        #----------------------------------------------------
        # HYBRID GTM 3: NJ-LogDet guide + NJ-LogDet subset trees
        #----------------------------------------------------
        # 1. Reuse the NJ-LogDet guide tree from Hybrid GTM 2
        if [ ! -f "$HYBRID_GTM3_DIR/guide_tree.tree" ]; then
            if [ -f "$HYBRID_GTM2_DIR/guide_tree.tree" ] && [ -s "$HYBRID_GTM2_DIR/guide_tree.tree" ]; then
                echo "    Copying NJ-LogDet guide tree for Hybrid GTM 3"
                cp "$HYBRID_GTM2_DIR/guide_tree.tree" "$HYBRID_GTM3_DIR/guide_tree.tree"
                # Record dependency
                mkdir -p "$HYBRID_GTM3_DIR"
                echo "nj_logdet" > "$HYBRID_GTM3_DIR/dependencies.txt"
            else
                echo "    ERROR: NJ-LogDet guide tree from Hybrid GTM 2 not found or empty"
                
                # Generate new NJ-LogDet guide tree if the one from Hybrid GTM 2 is not available
                echo "    Generating new NJ-LogDet guide tree for Hybrid GTM 3"
                if [ -f "$PHYLIP_FILE" ] && [ -s "$PHYLIP_FILE" ]; then
                    measure_performance "hybrid_gtm3_guide" "\"$FASTME_FILEPATH\" -i \"$PHYLIP_FILE\" -o \"$HYBRID_GTM3_DIR/guide_tree.tree\" -mN -dL" "$MODEL_COND" "$REP"
                    
                    # Check if guide tree was created successfully
                    if [ ! -s "$HYBRID_GTM3_DIR/guide_tree.tree" ]; then
                        echo "    ERROR: FastME failed to create a valid guide tree for Hybrid GTM 3"
                        continue
                    fi
                else
                    echo "    ERROR: PHYLIP file not found for creating NJ-LogDet guide tree"
                    continue
                fi
            fi
        fi
        
        # 2. Decompose dataset using NJ-LogDet guide tree
        if [ ! -d "$HYBRID_GTM3_DIR/subsets" ] || [ $(ls -1 "$HYBRID_GTM3_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
            echo "    Decomposing dataset using NJ-LogDet guide tree for Hybrid GTM 3"
            mkdir -p "$HYBRID_GTM3_DIR/subsets"
            measure_performance "hybrid_gtm3_decompose" "$PYTHON_PATH $SCRIPTS_DIR/decompose.py --input-tree \"$HYBRID_GTM3_DIR/guide_tree.tree\" --sequence-file \"$ALIGNMENT\" --output-prefix \"$HYBRID_GTM3_DIR/subsets/subset\" --maximum-size 250 --mode \"centroid\"" "$MODEL_COND" "$REP"
            
            # Check if decomposition was successful
            if [ $(ls -1 "$HYBRID_GTM3_DIR/subsets/subset"*.out 2>/dev/null | wc -l) -eq 0 ]; then
                echo "    ERROR: Failed to decompose dataset for Hybrid GTM 3"
                continue
            else
                echo "    Successfully decomposed dataset for Hybrid GTM 3"
            fi
        fi
        
        # 3. Run NJ-LogDet on each subset
        SUBSET_COUNT=$(ls -1 "$HYBRID_GTM3_DIR/subsets/subset"*.out 2>/dev/null | wc -l)
        echo "    Found $SUBSET_COUNT subsets for Hybrid GTM 3"
 
        if [ "$SUBSET_COUNT" -gt 0 ]; then
            echo "    Running NJ-LogDet on subsets for Hybrid GTM 3"
            for SUBSET_FILE in "$HYBRID_GTM3_DIR/subsets/subset"*.out; do
                SUBSET_PREFIX="${SUBSET_FILE%.out}"
                SUBSET_TREE="$SUBSET_PREFIX.tree"
                SUBSET_PHY="$SUBSET_PREFIX.phy"
                SUBSET_NUM=$(basename "$SUBSET_PREFIX" | sed 's/subset//')
                
                if [ ! -f "$SUBSET_TREE" ] || [ ! -s "$SUBSET_TREE" ]; then
                    echo "      Processing subset $(basename "$SUBSET_FILE")"
                    
                    # Convert to PHYLIP
                    measure_performance "hybrid_gtm3_convert_phy_$SUBSET_NUM" "$PYTHON_PATH $SCRIPTS_DIR/convert_subset_to_phylip.py \"$SUBSET_FILE\" \"$SUBSET_PHY\"" "$MODEL_COND" "$REP"
                    
                    # Check if PHYLIP conversion was successful
                    if [ ! -f "$SUBSET_PHY" ] || [ ! -s "$SUBSET_PHY" ]; then
                        echo "      ERROR: Failed to convert subset to PHYLIP format"
                        continue
                    fi
                    
                    # Run FastME with NJ and LogDet
                    measure_performance "hybrid_gtm3_subset_tree_$SUBSET_NUM" "\"$FASTME_FILEPATH\" -i \"$SUBSET_PHY\" -o \"$SUBSET_TREE\" -mN -dL" "$MODEL_COND" "$REP"
                    
                    # Check if tree was created successfully
                    if [ ! -s "$SUBSET_TREE" ]; then
                        echo "      ERROR: FastME failed to create a valid tree for subset"
                    else
                        echo "      Successfully created tree for subset"
                    fi
                fi
            done
            
            # Check if all subset trees were created
            TREE_COUNT=$(ls -1 "$HYBRID_GTM3_DIR/subsets/subset"*.tree 2>/dev/null | wc -l)
            if [ "$TREE_COUNT" -ne "$SUBSET_COUNT" ]; then
                echo "    ERROR: Not all subset trees were created for Hybrid GTM 3"
                echo "    Found $TREE_COUNT trees for $SUBSET_COUNT subsets"
            fi
        fi
        
        # 4. Merge subset trees using GTM
        if [ "$SUBSET_COUNT" -gt 0 ] && [ ! -f "$HYBRID_GTM3_DIR/merged_tree.tree" ]; then
            echo "    Merging subset trees using GTM (Hybrid GTM 3)"
            
            # Create list of subset trees
            SUBSET_TREES=()
            for SUBSET_TREE in "$HYBRID_GTM3_DIR/subsets/subset"*.tree; do
                if [ -f "$SUBSET_TREE" ] && [ -s "$SUBSET_TREE" ]; then
                    SUBSET_TREES+=("$SUBSET_TREE")
                fi
            done
            
            # Check if we have subset trees to merge
            if [ ${#SUBSET_TREES[@]} -gt 0 ]; then
                # Run GTM - create a string with all subset trees
                SUBSET_TREE_STR="${SUBSET_TREES[@]}"
                measure_performance "hybrid_gtm3_merge" "$PYTHON_PATH $SCRIPTS_DIR/gtm.py -s \"$HYBRID_GTM3_DIR/guide_tree.tree\" -t $SUBSET_TREE_STR -o \"$HYBRID_GTM3_DIR/merged_tree.tree\" -m \"convex\"" "$MODEL_COND" "$REP"
                
                # Check if merged tree was created successfully
                if [ ! -s "$HYBRID_GTM3_DIR/merged_tree.tree" ]; then
                    echo "    ERROR: GTM failed to create a valid merged tree for Hybrid GTM 3"
                else
                    echo "    Successfully created merged tree for Hybrid GTM 3"
                fi
            else
                echo "    ERROR: No valid subset trees found for merging in Hybrid GTM 3"
            fi
        fi
        
        #----------------------------------------------------
        # Compare all trees to the true tree
        #----------------------------------------------------
        
        # Check if treecompare.py can run properly
        echo "Testing dendropy for tree comparison:"
        $PYTHON_PATH -c "
import sys
print(f'Python path: {sys.path}')
try:
    import dendropy
    print(f'Successfully imported dendropy version: {dendropy.__version__}')
    from dendropy.calculate import treecompare
    print('Successfully imported treecompare module')
except ImportError as e:
    print(f'Failed to import dendropy: {e}')
    print('Attempting to install dendropy...')
    import subprocess
    subprocess.check_call(['pip', 'install', 'dendropy'])
    import dendropy
    print(f'Successfully installed and imported dendropy version: {dendropy.__version__}')
    from dendropy.calculate import treecompare
    print('Successfully imported treecompare module')
"
        
        # Compare baseline FastTree GTR to true tree
        if [ -f "$BASELINE_DIR/fasttree_gtr.tree" ] && [ ! -f "$BASELINE_DIR/fasttree_gtr_comparison.txt" ]; then
            echo "    Comparing Baseline FastTree GTR to true tree"
            measure_performance "compare_fasttree_gtr" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/fasttree_gtr.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/fasttree_gtr_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/fasttree_gtr_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline FastTree GTR to true tree"
            fi
        fi
        
        # Compare baseline FastTree JC to true tree
        if [ -f "$BASELINE_DIR/fasttree_jc.tree" ] && [ ! -f "$BASELINE_DIR/fasttree_jc_comparison.txt" ]; then
            echo "    Comparing Baseline FastTree JC to true tree"
            measure_performance "compare_fasttree_jc" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/fasttree_jc.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/fasttree_jc_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/fasttree_jc_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline FastTree JC to true tree"
            fi
        fi
        
        # Compare baseline NJ JC to true tree
        if [ -f "$BASELINE_DIR/nj_jc.tree" ] && [ ! -f "$BASELINE_DIR/nj_jc_comparison.txt" ]; then
            echo "    Comparing Baseline NJ JC to true tree"
            measure_performance "compare_nj_jc" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/nj_jc.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/nj_jc_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/nj_jc_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline NJ JC to true tree"
            fi
        fi
        
        # Compare baseline NJ LogDet to true tree
        if [ -f "$BASELINE_DIR/nj_logdet.tree" ] && [ ! -f "$BASELINE_DIR/nj_logdet_comparison.txt" ]; then
            echo "    Comparing Baseline NJ LogDet to true tree"
            measure_performance "compare_nj_logdet" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/nj_logdet.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/nj_logdet_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/nj_logdet_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline NJ LogDet to true tree"
            fi
        fi
        
        # Compare baseline NJ p-distance to true tree
        if [ -f "$BASELINE_DIR/nj_pdist.tree" ] && [ ! -f "$BASELINE_DIR/nj_pdist_comparison.txt" ]; then
            echo "    Comparing Baseline NJ p-distance to true tree"
            measure_performance "compare_nj_pdist" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/nj_pdist.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/nj_pdist_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/nj_pdist_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline NJ p-distance to true tree"
            fi
        fi
        
        # Compare baseline FastME BME to true tree
        if [ -f "$BASELINE_DIR/fastme_bme.tree" ] && [ ! -f "$BASELINE_DIR/fastme_bme_comparison.txt" ]; then
            echo "    Comparing Baseline FastME BME to true tree"
            measure_performance "compare_fastme_bme" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$BASELINE_DIR/fastme_bme.tree\" \"$TRUE_TREE\" \"$BASELINE_DIR/fastme_bme_comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$BASELINE_DIR/fastme_bme_comparison.txt" ]; then
                echo "    ERROR: Failed to compare Baseline FastME BME to true tree"
            fi
        fi
        
        # Compare Original GTM result to true tree
        if [ -f "$ORIGINAL_GTM_DIR/merged_tree.tree" ] && [ ! -f "$ORIGINAL_GTM_DIR/comparison.txt" ]; then
            echo "    Comparing Original GTM to true tree"
            measure_performance "compare_original_gtm" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$ORIGINAL_GTM_DIR/merged_tree.tree\" \"$TRUE_TREE\" \"$ORIGINAL_GTM_DIR/comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$ORIGINAL_GTM_DIR/comparison.txt" ]; then
                echo "    ERROR: Failed to compare Original GTM to true tree"
            fi
        fi
        
        # Compare Hybrid GTM 1 result to true tree
        if [ -f "$HYBRID_GTM1_DIR/merged_tree.tree" ] && [ ! -f "$HYBRID_GTM1_DIR/comparison.txt" ]; then
            echo "    Comparing Hybrid GTM 1 to true tree"
            measure_performance "compare_hybrid_gtm1" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$HYBRID_GTM1_DIR/merged_tree.tree\" \"$TRUE_TREE\" \"$HYBRID_GTM1_DIR/comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$HYBRID_GTM1_DIR/comparison.txt" ]; then
                echo "    ERROR: Failed to compare Hybrid GTM 1 to true tree"
            fi
        fi
        
        # Compare Hybrid GTM 2 result to true tree
        if [ -f "$HYBRID_GTM2_DIR/merged_tree.tree" ] && [ ! -f "$HYBRID_GTM2_DIR/comparison.txt" ]; then
            echo "    Comparing Hybrid GTM 2 to true tree"
            measure_performance "compare_hybrid_gtm2" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$HYBRID_GTM2_DIR/merged_tree.tree\" \"$TRUE_TREE\" \"$HYBRID_GTM2_DIR/comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$HYBRID_GTM2_DIR/comparison.txt" ]; then
                echo "    ERROR: Failed to compare Hybrid GTM 2 to true tree"
            fi
        fi
        
        # Compare Hybrid GTM 3 result to true tree
        if [ -f "$HYBRID_GTM3_DIR/merged_tree.tree" ] && [ ! -f "$HYBRID_GTM3_DIR/comparison.txt" ]; then
            echo "    Comparing Hybrid GTM 3 to true tree"
            measure_performance "compare_hybrid_gtm3" "$PYTHON_PATH $SCRIPTS_DIR/treecompare.py \"$HYBRID_GTM3_DIR/merged_tree.tree\" \"$TRUE_TREE\" \"$HYBRID_GTM3_DIR/comparison.txt\"" "$MODEL_COND" "$REP"
            
            # Check if comparison was successful
            if [ ! -f "$HYBRID_GTM3_DIR/comparison.txt" ]; then
                echo "    ERROR: Failed to compare Hybrid GTM 3 to true tree"
            fi
        fi
    done
done

#----------------------------------------------------
# Summarize results
#----------------------------------------------------
echo "Creating summary results"

# Check if there are comparison files to process
if [ $(find "$RESULTS_DIR" -name "*comparison.txt" | wc -l) -eq 0 ]; then
    echo "WARNING: No comparison files found. Skipping summary stage."
    exit 0
fi

# Check if summarize_results.py can run properly
echo "Testing pandas and matplotlib for results summarization:"
$PYTHON_PATH -c "
try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print('Successfully imported pandas, numpy, and matplotlib')
except ImportError as e:
    print(f'Failed to import required packages: {e}')
    print('Attempting to install required packages...')
    import subprocess
    subprocess.check_call(['pip', 'install', 'pandas', 'numpy', 'matplotlib'])
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print('Successfully installed and imported pandas, numpy, and matplotlib')
"

# Create summary for baseline methods
echo "Creating baseline summary"
$PYTHON_PATH $SCRIPTS_DIR/summarize_results.py \
    --input-dir "$RESULTS_DIR/baseline" \
    --method-patterns "fasttree_gtr:FastTree (GTR),fasttree_jc:FastTree (JC),nj_jc:NJ (JC),nj_logdet:NJ (LogDet),nj_pdist:NJ (p-distance),fastme_bme:FastME (BME)" \
    --output-prefix "$RESULTS_DIR/summary/baseline"

# Check if baseline summary was created successfully
if [ ! -f "$RESULTS_DIR/summary/baseline_raw.csv" ] || [ ! -f "$RESULTS_DIR/summary/baseline_summary.csv" ]; then
    echo "WARNING: Failed to create baseline summary files"
fi

# Only run GTM summary if there are GTM results
GTM_RESULTS_COUNT=$(find "$RESULTS_DIR/original_gtm" "$RESULTS_DIR/hybrid_gtm1" "$RESULTS_DIR/hybrid_gtm2" "$RESULTS_DIR/hybrid_gtm3" -name "comparison.txt" 2>/dev/null | wc -l)

if [ "$GTM_RESULTS_COUNT" -gt 0 ]; then
    echo "Creating GTM summary (found $GTM_RESULTS_COUNT comparison files)"
    
    # Create summary for all GTM methods
    $PYTHON_PATH $SCRIPTS_DIR/summarize_gtm_results.py \
        --original-gtm-dir "$RESULTS_DIR/original_gtm" \
        --hybrid-gtm1-dir "$RESULTS_DIR/hybrid_gtm1" \
        --hybrid-gtm2-dir "$RESULTS_DIR/hybrid_gtm2" \
        --hybrid-gtm3-dir "$RESULTS_DIR/hybrid_gtm3" \
        --output-prefix "$RESULTS_DIR/summary/gtm"
    
    # Check if GTM summary was created successfully
    if [ ! -f "$RESULTS_DIR/summary/gtm_raw.csv" ] || [ ! -f "$RESULTS_DIR/summary/gtm_summary.csv" ]; then
        echo "WARNING: Failed to create GTM summary files"
    else
        echo "Successfully created GTM summary files"
    fi
    
    # Create final comparison plots only if both summary files exist
    if [ -f "$RESULTS_DIR/summary/baseline_summary.csv" ] && [ -f "$RESULTS_DIR/summary/gtm_summary.csv" ]; then
        echo "Creating comparison plots"
        $PYTHON_PATH $SCRIPTS_DIR/create_comparison_plots.py \
            --baseline-summary "$RESULTS_DIR/summary/baseline_summary.csv" \
            --gtm-summary "$RESULTS_DIR/summary/gtm_summary.csv" \
            --output-dir "$RESULTS_DIR/summary"
        
        # Check if plots were created
        if [ $(find "$RESULTS_DIR/summary" -name "*.png" | wc -l) -eq 0 ]; then
            echo "WARNING: No plot files were created"
        else
            echo "Successfully created comparison plots"
        fi
    else
        echo "WARNING: Missing summary CSV files. Skipping comparison plots."
    fi
else
    echo "WARNING: No GTM comparison files found. Skipping GTM summary."
fi

# Add the analyze_performance.py script call at the end
echo "Analyzing performance data..."
$PYTHON_PATH $SCRIPTS_DIR/analyze_performance.py \
    --perf-dir "$RESULTS_DIR/performance" \
    --output-dir "$RESULTS_DIR/summary"

# Create combined plots if accuracy data is available
if [ -f "$RESULTS_DIR/summary/all_methods_summary.csv" ]; then
    echo "Creating performance vs accuracy tradeoff plots..."
    $PYTHON_PATH $SCRIPTS_DIR/analyze_performance.py \
        --perf-dir "$RESULTS_DIR/performance" \
        --output-dir "$RESULTS_DIR/summary" \
        --accuracy-file "$RESULTS_DIR/summary/all_methods_summary.csv"
fi

echo "Analysis completed. Results are in $RESULTS_DIR"