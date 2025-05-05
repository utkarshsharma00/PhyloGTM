#!/usr/bin/env python
# scripts/verify_guide_tree.py

import dendropy
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python verify_guide_tree.py tree_file")
        sys.exit(1)
    
    tree_file = sys.argv[1]
    
    try:
        tree = dendropy.Tree.get(path=tree_file, schema='newick')
        print(f'Guide tree is valid with {len(tree.leaf_nodes())} taxa')
        sys.exit(0)
    except Exception as e:
        print(f'Invalid guide tree: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()