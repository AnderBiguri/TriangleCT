function tree=checkTree(tree)


tree.bin_n_elements=int32(tree.bin_n_elements);
tree.M=int32(tree.M);
tree.m=int32(tree.m);
tree.depth=int32(tree.depth);

tree.bin_box=tree.bin_box.';
tree.bin_box=tree.bin_box(:);
tree.bin_elements=tree.bin_elements.';
tree.bin_elements=tree.bin_elements(:);
tree.root=int32(tree.root);

if ispc
tree.bin_elements=int32(tree.bin_elements);
else
tree.bin_elements=int64(tree.bin_elements);

end

end