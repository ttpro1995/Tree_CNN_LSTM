# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.gold_label = None # node label for SST
        self.output = None # output node for SST
        self.nodes = None
        self._depth = None

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def set_spans(self):
        if self.num_children == 0:
            self.lo, self.hi = self.idx, self.idx
        else:
            for child in self.children:
                child.set_spans()
            self.lo = self.children[0].lo
            self.hi = self.children[0].hi
            for child in self.children:
                self.lo = min(self.lo, child.lo)
                self.hi = max(self.hi, child.hi)

    def depth(self):
        if self._depth is not None:
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def depth_first_preorder(self):
        "return list of subtree and itself"
        if self.nodes == None:
            nodes = []
            depth_first_preorder(self, nodes=nodes)
            self.nodes = nodes
        return self.nodes


def depth_first_preorder(tree, nodes):
    """
    :param tree: Tree object
    :param nodes: list of Tree object
    :return: 
    """
    if tree==None:
        return
    nodes.append(tree) # root at index 0
    if tree.num_children == 0:
        depth_first_preorder(None, nodes)
    else:
        for child in tree.children:
            depth_first_preorder(child, nodes)