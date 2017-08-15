import numpy as np

def generateAddition(n):
    """
    create addition problems with answers
    :param n: the number of questions to generate
    :return: a list of (Q, A) tuples
    """
    d = []
    for i in range(n):
        x = round(np.random.uniform(0, 50), 2)
        y = round(np.random.uniform(0, 50), 2)
        q = "Add " + repr(x) + " and " + repr(y)
        a = x + y
        d.append((q, a))
    return d

def generateSubtraction(n):
    """
        create subtraction problems with answers
        :param n: the number of questions to generate
        :return: a list of (Q, A) tuples
    """
    d = []
    for i in range(n):
        x = round(np.random.uniform(0, 50), 2)
        y = round(np.random.uniform(0, 100), 2)
        q = "Subtract " + repr(x) + " from " + repr(y)
        a = round(y - x, 2)
        d.append((q, a))
    return d

def generateMultiplication(n):
    """
        create multiplication problems with answers
        :param n: the number of questions to generate
        :return: a list of (Q, A) tuples
    """
    d = []
    for i in range(n):
        x = round(np.random.uniform(0, 10), 2)
        y = round(np.random.uniform(0, 10), 2)
        q = "Multiply " + repr(x) + " and " + repr(y)
        a = round(x*y, 2)
        d.append((q, a))
    return d

def generateDivision(n):
    """
        create division problems with answers
        :param n: the number of questions to generate
        :return: a list of (Q, A) tuples
    """
    d = []
    for i in range(n):
        x = round(np.random.uniform(0, 100), 2)
        y = round(np.random.uniform(1, 10), 2)
        q = "Divide " + repr(x) + " by " + repr(y)
        a = round(x/y, 2)
        d.append((q, a))
    return d
#print(generateAddition(2))
#print(generateSubtraction(10))
#print(generateMultiplication(4))
#print(generateDivision(5))

#print(generateDivision(5) + generateMultiplication(5))
