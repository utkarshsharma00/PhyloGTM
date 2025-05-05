# scripts/gtm_old.py
'''
Created on Nov 29, 2020

@author: Vlad
'''

import treeutils
import dendropy

def gtmMerge(constraintTrees, startTree):
    nameSpace = startTree.taxon_namespace
    startTree.is_rooted = False
    startTree.resolve_polytomies(limit=2)
    startTree.collapse_basal_bifurcation()
    startTree.update_bipartitions()
    
    splitSetMap = {}
    ctMap = {}
    for tree in constraintTrees:
        tree.migrate_taxon_namespace(nameSpace)
        tree.is_rooted = False
        tree.resolve_polytomies(limit=2)
        tree.collapse_basal_bifurcation()
        tree.update_bipartitions()

        leafset = tree.seed_node.bipartition.tree_leafset_bitmask
        splitSetMap[leafset] = set([b.split_bitmask for b in tree.bipartition_encoding])
        ctMap[leafset] = tree
    
    toRemove = []
    for e in startTree.preorder_edge_iter():
        if e.tail_node == None:
            continue
        
        split = e.bipartition.split_bitmask
        e.divider = True
        
        for leafset in splitSetMap:
            masks = splitSetMap[leafset]                  
            if split & leafset not in masks and ~split & leafset not in masks: 
                toRemove.append(e.head_node)
                break
            if split & leafset != 0 and split & leafset != leafset:
                if not e.divider:
                    toRemove.append(e.head_node)
                    break
                e.divider = False 
                e.leafset = leafset      
                        
    treeutils.collapseEdges(toRemove)    
    result = rejoin(startTree, ctMap)
    return result


def rejoin(tree, ctMap):  
    treeleaves = tree.seed_node.tree_leafset_bitmask    
    if treeleaves in ctMap:
        return ctMap[treeleaves]
    
    if len(tree.seed_node.child_nodes()) == 2:
        tree.collapse_basal_bifurcation()

    for n in tree.preorder_internal_node_iter():
        childs = n.child_nodes()
        if childs[0].edge.divider:
            newGroup = [childs[0]]
        else:
            newGroup = []
            if n.edge.tail_node != None:
                leafs =  n.edge.leafset
            else:
                leafs = childs[0].edge.leafset
            
            for child in childs:
                if child.edge.divider:
                    newGroup = [child]
                    break
                elif child.edge.leafset != leafs:
                    newGroup.append(child)
                  
        if len(newGroup) > 0:
                  
            t2l = 0
            for child in newGroup:
                t2l = t2l | child.edge.bipartition.leafset_bitmask
                n.remove_child(child)                
            t1l = treeleaves ^ t2l
            
            if len(newGroup) == 1:
                newNode = newGroup[0]
            else:
                newNode = dendropy.Node()
                for child in newGroup:
                    newNode.add_child(child)                    
                       
            a1 = n.child_edges()[0].bipartition.leafset_bitmask            
            a2 = newNode.child_edges()[0].bipartition.leafset_bitmask 

            t2 = dendropy.Tree(seed_node=newNode, taxon_namespace = tree.taxon_namespace)       
            t2.seed_node.tree_leafset_bitmask = t2l     

            tree.suppress_unifurcations()            
            tree.seed_node.tree_leafset_bitmask = t1l
                          
            r1 = rejoin(tree, ctMap)
            r2 = rejoin(t2, ctMap)            
            
            for e in r1.edges():
                if e.bipartition.leafset_bitmask == a1 or e.bipartition.leafset_bitmask == a1 ^ r1.seed_node.tree_leafset_bitmask:
                    n1 = e.tail_node.new_child()
                    n1.add_child(e.tail_node.remove_child(e.head_node))
                    break
            for e in r2.edges():
                if e.bipartition.leafset_bitmask == a2 or e.bipartition.leafset_bitmask == a2 ^ r2.seed_node.tree_leafset_bitmask:
                    n2 = e.tail_node.new_child()
                    n2.add_child(e.tail_node.remove_child(e.head_node))
                    break   

            r2.reseed_at(n2,update_bipartitions=False,
            collapse_unrooted_basal_bifurcation=False,
            suppress_unifurcations=False)
            n1.add_child(n2)
            
            r1.update_bipartitions()
            return r1