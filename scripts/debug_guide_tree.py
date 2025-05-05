#!/usr/bin/env python
# scripts/debug_guide_tree.py

import sys
import os
import dendropy

def main():
    if len(sys.argv) != 3:
        print("Usage: python debug_guide_tree.py guide_tree_path alignment_path")
        sys.exit(1)
    
    guide_tree_path = sys.argv[1]
    alignment_path = sys.argv[2]
    
    print(f'Guide tree path: {guide_tree_path}')
    print(f'File exists: {os.path.exists(guide_tree_path)}')
    print(f'File size: {os.path.getsize(guide_tree_path) if os.path.exists(guide_tree_path) else 0} bytes')
    print(f'Alignment path: {alignment_path}')
    print(f'Alignment exists: {os.path.exists(alignment_path)}')
    
    try:
        tree = dendropy.Tree.get(path=guide_tree_path, schema='newick')
        print(f'Successfully loaded tree with {len(tree.leaf_nodes())} leaf nodes')
        sys.exit(0)
    except Exception as e:
        print(f'Error loading tree: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()