import unittest
from collections import defaultdict, Counter
from pyparsing import (Forward, Word, alphas, alphanums,
                       nums, ZeroOrMore, Literal, Group,
                       ParseResults, Empty)


class Compute(object):
    """
    """
    memo = defaultdict(dict)

    @staticmethod
    def format_args(aStr):
        identifier = Word(alphas, alphanums + "_")
        integer = Word(nums)
        functor = identifier
        lparen = Literal("(").suppress()
        rparen = Literal(")").suppress()
        expression = Forward()
        arg = Group(expression) | identifier | integer | Empty()
        args = arg + ZeroOrMore("," + arg)
        expression << functor + Group(lparen + args + rparen)
        return expression.parseString(aStr)

    @staticmethod
    def select_func(leaves):
        count = Counter([i[0] for i in leaves])
        func = count.most_common(1)[0][0]
        argSet = set([tuple(i[1]) for i in leaves if i[0] == func])
        args = [list(i) for i in argSet]
        return func, args

    @classmethod
    def serial(cls, computations):
        output = []
        for computation in computations:
            parsed_args = cls.format_args(computation)
            output.append(cls.serial_eval(parsed_args))
        return output

    @classmethod
    def serial_eval(cls, parsed_args):
        f, args = parsed_args[0], parsed_args[1]
        parsed = []
        for i in args:
            if isinstance(i, ParseResults):
                parsed.append(cls.serial_eval(i))
            elif i.isdigit():
                parsed.append(int(i))
        return cls.funcs[f](parsed)

    @classmethod
    def batch(cls, computations):
        trees = {ind: Tree(cls.format_args(i))
                 for ind, i in enumerate(computations)}
        return cls.batch_eval(trees)

    @classmethod
    def batch_eval(cls, trees):
        if all([i.root.val is not None for i in trees.itervalues()]):
            return [trees[i].root.val for i in sorted(trees)]
        subtrees = [i for i in trees.itervalues() if not i.root.val]
        leaves = set([])

        for tree in subtrees:
            leaves = leaves.union(tree.collect_leaves())

        func, args = cls.select_func(leaves)
        cls.eval_leaves(func, args)

        for tree in subtrees:
            tree.prune(cls.memo)

        return cls.batch_eval(trees)

    @classmethod
    def eval_leaves(cls, func, args):
        args = [[int(j) for j in i] for i in args]
        batched = cls.funcs[func](args)
        for key, value in zip(args, batched):
            cls.memo[func][tuple(key)] = value

    @classmethod
    def compute(cls, computations=['f(g(h(2,3),5),g(g(3),h(4)),10)'],
                functionRegistry={'f': sum, 'g': sum, 'h': max},
                evalType='serial'):
        cls.funcs = functionRegistry
        return cls.__dict__[evalType].__func__(cls, computations)


class Tree(object):
    """
    """
    def __init__(self, parsed_args):
        self.root = Node(parsed_args, None)

    def collect_leaves(self, node=None, leaves=None):
        """
        """
        if not leaves:
            leaves = set([])
        if not node:
            node = self.root

        if not node.is_leaf():
            print node.f, node.children
            for child in node.children:
                if isinstance(child, Node):
                    leaves = self.collect_leaves(child, leaves)
        else:
            leaf = tuple([node.f, tuple(node.children)])
            leaves.add(leaf)
        return leaves

    def prune(self, known, node=None):
        if not node:
            node = self.root
        for child in node.children:
            if isinstance(child, Node):
                self.prune(known, child)
        if all([not node.children, node.is_leaf(), () in known[node.f]]):
            node.val = known[node.f][()]
            if node.parent:
                node.parent.children[:] = [i.val if isinstance(i, Node) and i.val is not None else i for i in node.parent.children]
        elif node.is_leaf() and tuple(int(i) for i in node.children) in known[node.f]:
            node.val = known[node.f][tuple(int(i) for i in node.children)]
            if node.parent:
                node.parent.children[:] = [i.val if isinstance(i, Node) and i.val is not None else i for i in node.parent.children]


class Node(object):
    """
    """
    def __init__(self, parsed_args, parent):
        f, args = parsed_args[0], parsed_args[1]
        self.f = f
        self.parent = parent
        self.children = []
        self.val = None
        for i in args:
            if isinstance(i, ParseResults):
                self.children.append(Node(i, self))
            elif i.isdigit():
                self.children.append(i)

    def is_leaf(self):
        return not any([isinstance(child, Node) for child in self.children]) or\
               not self.children


class TestFuncConj(unittest.TestCase):

    def test_serial_kwarg(self):
        result = Compute.compute()
        self.assertEqual(result, [25])

    def test_serial_zero(self):
        result = Compute.compute(['f()'])
        self.assertEqual(result, [0])

    def test_serial_zeros(self):
        result = Compute.compute(['f()', 'g()'])
        self.assertEqual(result, [0, 0])


def batch_sum(lstlst):
    output = []
    for i in lstlst:
        output.append(sum(i))
    return output


def batch_max(lstlst):
    output = []
    for i in lstlst:
        output.append(max(i))
    return output

if __name__ == "__main__":
    unittest.main()
