import unittest
from collections import defaultdict, Counter
from pyparsing import (Forward, Word, alphas, alphanums,
                       nums, ZeroOrMore, Literal, Group,
                       ParseResults)


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
        arg = Group(expression) | identifier | integer | ""
        args = arg + ZeroOrMore("," + arg)
        expression << functor + Group(lparen + args + rparen)
        return expression.parseString(aStr)

    @staticmethod
    def select_func(leaves):
        count = Counter([i[0] for i in leaves])
        return count.most_common(1)[0][0]

    @classmethod
    def serial(cls, computations, funcs):
        output = []
        for computation in computations:
            parsed_args = cls.format_args(computation)
            output.append(cls.serial_eval(parsed_args, funcs))
        return output

    @classmethod
    def serial_eval(cls, parsed_args, funcs):
        f, args = parsed_args[0], parsed_args[1]
        print f, args
        parsed = []
        for i in args:
            if isinstance(i, ParseResults):
                parsed.append(cls.serial_eval(i, funcs))
            elif i.isdigit():
                parsed.append(int(i))
        return cls.funcs[f](parsed)

    @classmethod
    def batch(cls, computations, funcs):
        trees = {ind: Tree(cls.format_args(i))
                 for ind, i in enumerate(computations)}
        cls.eval_batch(trees, funcs)

    @classmethod
    def batch_eval(cls, trees, funcs):

        if not any([i.val for i in trees]):
            return [trees(i).val for i in sorted(trees, key=lambda x:x)]

        subtrees = [i for i in trees if not i.val]
        leaves = set([])
        for tree in subtrees:
            leaves = leaves.union(tree.collect_leaves())

        func = cls.select_func(leaves)
        cls.eval_leaves(func, leaves, funcs[func])

        for tree in subtrees:
            leaves = tree.prune(leaves, cls.memo)
        return cls.batch_eval(trees, funcs)

    @classmethod
    def eval_leaves(cls, func, leaves, f):
        for oper, arg in leaves:
            if oper == func:
                cls.memo[func][arg] = f(arg)

    @classmethod
    def compute(cls, computations=['f(g(h(2,3),5),g(g(3),h(4)),10)'],
                functionRegistry={'f': sum, 'g': sum, 'h': max},
                evalType='serial'):
        cls.funcs = functionRegistry
        return cls.__dict__[evalType].__func__(cls, computations,
                                               functionRegistry)


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

        if node.is_leaf():
            print node.f, node.children
            for child in node.children:
                if isinstance(child, Node):
                    leaves = self.collect_leaves(child, leaves)
        else:
            leaf = tuple([node.f, tuple(node.children)])
            leaves.add(leaf)
        return leaves


class Node(object):
    """
    """
    def __init__(self, parsed_args, parent):
        f, args = parsed_args[0], parsed_args[1]
        self.f = f
        self.parent = parent
        self.children = []
        for i in args:
            if isinstance(i, ParseResults):
                self.children.append(Node(i, self))
            elif i.isdigit():
                self.children.append(i)

    def is_leaf(self):
        return any([isinstance(child, Node) for child in self.children]) or\
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

if __name__ == "__main__":
    unittest.main()
