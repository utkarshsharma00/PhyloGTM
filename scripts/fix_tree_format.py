#!/usr/bin/env python
# scripts/fix_tree_format.py

import dendropy
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python fix_tree_format.py tree_file")
        sys.exit(1)
    
    tree_file = sys.argv[1]
    
    try:
        tree = dendropy.Tree.get(path=tree_file, schema='newick', rooting='force-rooted')
        tree.write(path=f"{tree_file}.fixed", schema='newick')
        print(f'Successfully fixed tree format for {tree_file}')
        sys.exit(0)
    except Exception as e:
        print(f'Error fixing tree format: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()