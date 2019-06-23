# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from collections import deque


class Symbol(object):
    def __init__(self, value):
        """
        A propositional symbol.

        @param value: The string value of a propositional symbol
        """
        self.value = value

    def __str__(self):
        return self.value.encode('utf-8')


class DefiniteClause(object):
    def __init__(self, premise=[], conclusion=None):
        """
        A definite clause: a disjunction of literals in which exactly one
        is positive.

        @param premises: An array of Symbols, representing a conjunction of
                         literals (default: [])
        @param conclusion: Symbol implied by the given premise (default: None)
        """

        self.premise = premise
        self.conclusion = conclusion

    def __str__(self):
        return (' ∧ '.join([str(p).decode('utf8') for p in self.premise]) +
                ' ⇒ {}'.format(self.conclusion)).encode('utf-8')


def forward_chaining(knowledge_base, query):
    """
    Implementation of the forward chaining algorithm as described in Fig 7.15
    of Russell, S. J., & Norvig, P. (2016).

    @param knowledge_base: An array of known DefiniteClauses
    @param query: A Symbol whose truth must be ascertained from knowledge_base

    @return: True if query is entailed by knowledge_base, False otherwise
    """

    unprocessed_truths = deque(clause.conclusion
                               for clause in knowledge_base
                               if not clause.premise)
    inferred = set()
    unknown_premises = {c: len(c.premise) for c in knowledge_base}

    while unprocessed_truths:
        item = unprocessed_truths.popleft()

        if item == query:
            return True

        if item not in inferred:
            inferred.add(item)
            kb_clauses_with_item = list(filter(lambda c: item in c.premise,
                                               knowledge_base))

            for clause in kb_clauses_with_item:
                unknown_premises[clause] -= 1

                if unknown_premises[clause] == 0:
                    unprocessed_truths.append(clause.conclusion)

    return False


# Now we test
print 'Russell & Norvig (2016) pg 259, Figure 7.16'
P = Symbol('P')
Q = Symbol('Q')
L = Symbol('L')
M = Symbol('M')
A = Symbol('A')
B = Symbol('B')

kb = [
    DefiniteClause([P], Q),
    DefiniteClause([L, M], P),
    DefiniteClause([B, L], M),
    DefiniteClause([A, P], L),
    DefiniteClause([A, B], L)
]
print 'A is: {}'.format('entailed'
                        if forward_chaining(kb, A)
                        else 'not entailed')
print 'B is: {}'.format('entailed'
                        if forward_chaining(kb, B)
                        else 'not entailed')

print '\n7.2 in-class work'
A = Symbol('A')
B = Symbol('B')
C = Symbol('C')
D = Symbol('D')
E = Symbol('E')
F = Symbol('F')
G = Symbol('G')
H = Symbol('H')
J = Symbol('J')
K = Symbol('K')

kb = [
    DefiniteClause([B, C], A),
    DefiniteClause([D], B),
    DefiniteClause([E], B),
    DefiniteClause([H], D),
    DefiniteClause([G, B], F),
    DefiniteClause([C, K], G),
    DefiniteClause([A, B], J),
    DefiniteClause(conclusion=C),
    DefiniteClause(conclusion=E)
]

print 'A is: {}'.format('entailed'
                        if forward_chaining(kb, A)
                        else 'not entailed')
print 'B is: {}'.format('entailed'
                        if forward_chaining(kb, B)
                        else 'not entailed')
print 'C is: {}'.format('entailed'
                        if forward_chaining(kb, C)
                        else 'not entailed')
print 'D is: {}'.format('entailed'
                        if forward_chaining(kb, D)
                        else 'not entailed')
print 'E is: {}'.format('entailed'
                        if forward_chaining(kb, E)
                        else 'not entailed')
print 'F is: {}'.format('entailed'
                        if forward_chaining(kb, F)
                        else 'not entailed')
print 'G is: {}'.format('entailed'
                        if forward_chaining(kb, G)
                        else 'not entailed')
print 'H is: {}'.format('entailed'
                        if forward_chaining(kb, H)
                        else 'not entailed')
print 'J is: {}'.format('entailed'
                        if forward_chaining(kb, J)
                        else 'not entailed')
print 'K is: {}'.format('entailed'
                        if forward_chaining(kb, K)
                        else 'not entailed')
