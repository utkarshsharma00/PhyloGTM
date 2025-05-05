# import dendropy

# def compareTreesFromPath(treePath1, treePath2):
#     print("Comparing {} with {}".format(treePath1, treePath2))
    
#     tax = dendropy.TaxonNamespace()
#     tr1 = dendropy.Tree.get(path=treePath1,
#                             schema='newick',
#                             rooting='force-unrooted',
#                             taxon_namespace=tax,
#                             preserve_underscores=True)
#     tr2 = dendropy.Tree.get(path=treePath2,
#                             schema='newick',
#                             rooting='force-unrooted',
#                             taxon_namespace=tax,
#                             preserve_underscores=True)

#     tr1.collapse_basal_bifurcation(set_as_unrooted_tree=True)
#     tr2.collapse_basal_bifurcation(set_as_unrooted_tree=True)

#     return compareDendropyTrees(tr1, tr2)
#     #print("RF distance on %d shared leaves: %d" % (nl, fp + fn))

# def compareDendropyTrees(tr1, tr2):
#     from dendropy.calculate.treecompare \
#         import false_positives_and_negatives

#     lb1 = set([l.taxon.label for l in tr1.leaf_nodes()])
#     lb2 = set([l.taxon.label for l in tr2.leaf_nodes()])
    
#     com = lb1.intersection(lb2)
#     if com != lb1 or com != lb2:
#         com = list(com)
#         tns = dendropy.TaxonNamespace(com)

#         tr1.retain_taxa_with_labels(com)
#         tr1.migrate_taxon_namespace(tns)

#         tr2.retain_taxa_with_labels(com)
#         tr2.migrate_taxon_namespace(tns)
#     com = list(com)

#     tr1.update_bipartitions()
#     tr2.update_bipartitions()

#     nl = len(com)
#     ei1 = len(tr1.internal_edges(exclude_seed_edge=True))
#     ei2 = len(tr2.internal_edges(exclude_seed_edge=True))

#     [fp, fn] = false_positives_and_negatives(tr1, tr2)
#     rf = float(fp + fn) / (ei1 + ei2)

#     return (nl, ei1, ei2, fp, fn, rf)
    
# #print("estimated tree error:", compareTreesFromPath('/media/ellie/easystore/BATCH-SCAMPP/testing/16S.B.ALL/0/backbone_epa.tree', '/media/ellie/easystore/BATCH-SCAMPP/testing/16S.B.ALL/0/truetopo.tree'))  


import sys
import dendropy

def main():
    """Compare two trees and output comparison metrics"""
    if len(sys.argv) != 4:
        print("Usage: python treecompare.py <estimated_tree> <true_tree> <output_file>")
        sys.exit(1)
    
    est_tree_path = sys.argv[1]
    true_tree_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Print input parameters for debugging
    print(f"Comparing trees: {est_tree_path} vs {true_tree_path}")
    
    try:
        # Create a common taxon namespace
        taxa = dendropy.TaxonNamespace()
        
        # Load trees with more flexible options
        true_tree = dendropy.Tree.get(
            path=true_tree_path,
            schema="newick",
            taxon_namespace=taxa,
            preserve_underscores=True,
            rooting="force-unrooted"
        )
        
        est_tree = dendropy.Tree.get(
            path=est_tree_path,
            schema="newick",
            taxon_namespace=taxa,
            preserve_underscores=True,
            rooting="force-unrooted"
        )
        
        # Handle potential basal bifurcation issues
        true_tree.collapse_basal_bifurcation(set_as_unrooted_tree=True)
        est_tree.collapse_basal_bifurcation(set_as_unrooted_tree=True)
        
        # Get common taxa
        true_labels = set(l.taxon.label for l in true_tree.leaf_nodes())
        est_labels = set(l.taxon.label for l in est_tree.leaf_nodes())
        common_labels = true_labels.intersection(est_labels)
        
        print(f"True tree has {len(true_labels)} taxa")
        print(f"Estimated tree has {len(est_labels)} taxa")
        print(f"Trees share {len(common_labels)} common taxa")
        
        # Ensure there are enough taxa for comparison
        if len(common_labels) < 3:
            print("ERROR: Less than 3 common taxa found, cannot compare trees")
            with open(output_path, 'w') as f:
                f.write("Tree comparison results:\n")
                f.write("ERROR: Less than 3 common taxa found, cannot compare trees\n")
                f.write(f"True tree taxa: {len(true_labels)}\n")
                f.write(f"Estimated tree taxa: {len(est_labels)}\n")
                f.write(f"Common taxa: {len(common_labels)}\n")
                f.write("RF distance: N/A\n")
                f.write("FN rate: N/A\n")
                f.write("FP rate: N/A\n")
            return
        
        # Prune trees to common taxa
        true_tree.retain_taxa_with_labels(common_labels)
        est_tree.retain_taxa_with_labels(common_labels)
        
        # Update bipartitions
        true_tree.update_bipartitions()
        est_tree.update_bipartitions()
        
        # Calculate metrics
        from dendropy.calculate.treecompare import false_positives_and_negatives
        
        # Counts of internal edges (excluding seed edge)
        true_edges = len(true_tree.internal_edges(exclude_seed_edge=True))
        est_edges = len(est_tree.internal_edges(exclude_seed_edge=True))
        
        # Get false positives and negatives
        fp, fn = false_positives_and_negatives(true_tree, est_tree)
        
        # Calculate Robinson-Foulds distance (sum of FP and FN)
        rf_dist = fp + fn
        
        # Calculate FN and FP rates
        fn_rate = fn / true_edges if true_edges > 0 else 0
        fp_rate = fp / est_edges if est_edges > 0 else 0
        
        # Print results
        print(f"RF distance: {rf_dist}")
        print(f"FN rate: {fn_rate}")
        print(f"FP rate: {fp_rate}")
        
        # Write results to output file
        with open(output_path, 'w') as f:
            f.write("Tree comparison results:\n")
            f.write(f"RF distance: {rf_dist}\n")
            f.write(f"FN rate: {fn_rate}\n")
            f.write(f"FP rate: {fp_rate}\n")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        # Write error to output file
        with open(output_path, 'w') as f:
            f.write("Tree comparison results:\n")
            f.write(f"ERROR: {str(e)}\n")
            f.write("RF distance: N/A\n")
            f.write("FN rate: N/A\n")
            f.write("FP rate: N/A\n")

if __name__ == "__main__":
    main()