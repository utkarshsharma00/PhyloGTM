#!/usr/bin/env python
# scripts/convert_subset_to_phylip.py

from Bio import AlignIO
import os
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_subset_to_phylip.py input_fasta output_phylip")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        AlignIO.convert(input_file, 'fasta', output_file, 'phylip-relaxed')
        print(f'Successfully converted {input_file} to PHYLIP format')
        sys.exit(0)
    except Exception as e:
        print(f'Error converting file: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()