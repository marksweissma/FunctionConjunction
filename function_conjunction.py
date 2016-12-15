import unittest
from collections import defaultdict, Counter
from pyparsing import (Forward, Word, alphas, alphanums,
                       nums, ZeroOrMore, Literal, Group,
                       ParseResults, Empty, Combine, Optional)


class Compute(object):
    """
    Class for computing composed functions
    """
    @staticmethod
    def _format_args(aStr):
        """
        Process composed function string into nested pyparsing.ParseResults

        :param str aStr: string to parse
        :return: formatting result
        :rtype: pyparsing.ParseResults
        """

        identifier = Word(alphas, alphanums + "_")
        integer = Combine(Optional(Literal('-')) + Word(nums))
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
        """
        Select function to evaluate and collect arguments

        :param set.tuple.(str, list.int) leaves: function argument pairs
        :return: FunctionRegisty keye
        :rtype: str
        :return arguments for function to operate on
        :rtype: list.list.int
        """
        count = Counter([i[0] for i in leaves])
        func = count.most_common(1)[0][0]
        argSet = set([tuple(i[1]) for i in leaves if i[0] == func])
        args = [list(i) for i in argSet]
        return func, args

    @classmethod
    def serial(cls, computations):
        """
        Evaluation of each computation in serial

        :param list.str computations: list of string computations to execute
        :return: computation results
        :rtype: list
        """
        output = []
        for computation in computations:
            parsed_args = cls._format_args(computation)
            output.append(cls._serial_eval(parsed_args))
        return output

    @classmethod
    def _serial_eval(cls, parsed_args):
        """
        Evaulate computation specified by parsed_args in serial

        :param pyparsing.ParseResults parsed_args: computation
        :return: function applied on args
        """

        f, args = parsed_args[0], parsed_args[1]
        parsed = []
        for i in args:
            if isinstance(i, ParseResults):
                parsed.append(cls._serial_eval(i))
            else:
                try:
                    parsed.append(int(i))
                except ValueError:
                    pass
        return cls.funcs[f](parsed)

    @classmethod
    def batch(cls, computations):
        """
        Evaluation of each computation in batch

        :param list.str computations: list of string computations to execute
        :return: computation results
        :rtype: list
        """
        trees = {ind: Tree(cls._format_args(i))
                 for ind, i in enumerate(computations)}
        output = cls._batch_eval(trees)

        if cls.verbose:
            print 'batch calls: ' + str(cls.batchCalls)
        return output

    @classmethod
    def _batch_eval(cls, trees):
        """
        Evaulate computations in each tree containing parsed arguments

        :param dict trees: dictionary of pyparsing.ParseResults computation
        :return: computation results
        :rtype: list
        """
        if all([i.root.val is not None for i in trees.itervalues()]):
            return [trees[i].root.val for i in sorted(trees)]
        subtrees = [i for i in trees.itervalues() if i.root.val is None]
        leaves = set([])

        for tree in subtrees:
            leaves = leaves.union(tree.collect_leaves())

        func, args = cls.select_func(leaves)
        cls._eval_leaves(func, args)

        for tree in subtrees:
            tree.prune(cls.memo)

        return cls._batch_eval(trees)

    @classmethod
    def _eval_leaves(cls, func, args):
        """
        Apply function to batch args, store results in cls.memo

        :param str func: key in FunctionRegistry for function
        :param list.list args: batch args
        """
        cls.batchCalls += 1
        args = [[int(j) for j in i] for i in args]
        batched = cls.funcs[func](args)
        for key, value in zip(args, batched):
            cls.memo[func][tuple(key)] = value

    @classmethod
    def compute(cls, computations=['f(g(h(2,3),5),g(g(3),h(4)),10)'],
                functionRegistry={'f': sum, 'g': sum, 'h': max},
                evalType='serial', verbose=False):
        """
        Evaluate computations using functions supplied in functionRegistry\
                with either serial or batch methods

        :param list.str computations: computations to evaluate
        :param dict functionRegistry: map strings in computations to functions
        :param str evalType: method for evaluation. serial requires functions\
                in registry to accept a list.int, batch requires functions\
                in registry to accept list.list.int
        :parm bool verbose: output additional information during run\
                currently only prints number of batch calls if batch evalType
        :return: evaluated computations
        :rtype: list
        """

        cls.funcs = functionRegistry
        cls.verbose = verbose
        cls.memo = defaultdict(dict)
        cls.batchCalls = 0
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

        # Empty argument case, these can be combined by
        # Modifying types in arg parse !!!TODO
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
            else:
                try:
                    self.children.append(int(i))
                except ValueError:
                    pass

    def is_leaf(self):
        return not any([isinstance(child, Node) for child in self.children])


class TestSerialFuncConj(unittest.TestCase):

    def test_kwarg(self):

        computations = ['f(g(h(2,3),5),g(g(3),h(4)),10)']
        functionRegistry = {'f': sum, 'g': sum, 'h': max},
        evalType = 'serial'
        result = Compute.compute(computations=computations,
                                 functionRegistry=functionRegistry,
                                 evalType=evalType)
        self.assertEqual(result, [25])

    def test_zero(self):
        result = Compute.compute(['f()'])
        self.assertEqual(result, [0])

    def test_zeros(self):
        result = Compute.compute(['f()', 'g()'])
        self.assertEqual(result, [0, 0])

    def test_nested_zero(self):
        result = Compute.compute(['f(g())'])
        self.assertEqual(result, [0])

    def test_nested_zeros(self):
        result = Compute.compute(['f(g())', 'f(g())'])
        self.assertEqual(result, [0, 0])

    def test_nested1(self):
        result = Compute.compute(['f(g(1))'])
        self.assertEqual(result, [1])


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
