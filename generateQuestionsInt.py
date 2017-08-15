import numpy as np


def generateAddition(n):
    """
    create n addition problems with answers
    :param n: the number of questions to generate
    :return: a list of tuples. Each tuple is in the form (Q, A)
    """
    d = []
    for i in range(n):
        x = np.random.randint(0, 50)
        y = np.random.randint(0, 50)
        q = "Add " + repr(x) + " and " + repr(y)
        a = x + y
        d.append((q, a))
    return d

def generateSubtraction(n):
    """
    create n subtraction problems with answers
    :param n: the number of questions to generate
    :return: a list of tuples. Each tuple is in the form (Q, A)
    """
    d = []
    for i in range(n):
        x = np.random.randint(0, 50)
        y = np.random.randint(0, 100)
        q = "Subtract " + repr(x) + " from " + repr(y)
        a = abs(y - x)
        d.append((q, a))
    return d

def generateMultiplication(n):
    """
        create n multiplication problems with answers
        :param n: the number of questions to generate
        :return: a list of tuples. Each tuple is in the form (Q, A)
    """
    d = []
    for i in range(n):
        x = np.random.randint(0, 10)
        y = np.random.randint(0, 10)
        q = "Multiply " + repr(x) + " and " + repr(y)
        a = x * y
        d.append((q, a))
    return d

def generateDivision(n):
    """
            create n division problems with answers
            :param n: the number of questions to generate
            :return: a list of tuples. Each tuple is in the form (Q, A)
    """
    d = []
    for i in range(n):
        x = np.random.randint(0, 100)
        y = np.random.randint(1, 10)
        q = "Divide " + repr(x) + " by " + repr(y)
        a = int(x/y)
        d.append((q, a))
    return d
#print(generateAddition(2))
#print(generateSubtraction(10))
#print(generateMultiplication(4))
#print(generateDivision(5))

#print(generateDivision(5) + generateMultiplication(5))
