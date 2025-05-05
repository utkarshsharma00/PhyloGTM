#!/usr/bin/env python
# scripts/convert_fasta_to_phylip.py

from Bio import AlignIO
import os
import sys

def convert_fasta_to_phylip(input_file, output_file):
    """Convert a FASTA file to PHYLIP format"""
    try:
        # Check if input file exists and has content
        if not os.path.exists(input_file):
            print(f'ERROR: Input file {input_file} does not exist')
            return False
        
        if os.path.getsize(input_file) == 0:
            print(f'ERROR: Input file {input_file} is empty')
            return False
        
        # Make sure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert file
        AlignIO.convert(input_file, 'fasta', output_file, 'phylip-relaxed')
        print(f"Successfully converted from FASTA to PHYLIP")
        return True
    except Exception as e:
        print(f'Error converting file: {e}')
        return False

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python convert_fasta_to_phylip.py input.fasta output.phy")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not convert_fasta_to_phylip(input_file, output_file):
        sys.exit(1)