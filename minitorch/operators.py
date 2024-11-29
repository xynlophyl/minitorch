"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Any

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiplies two numbers
    
    Args:
        x: float 
        y: float

    Returns:
        product of x and y

    """
    return x*y

def id(x: Any) -> Any:
    """Returns input unchanged
    
    Args:
        x: float 

    Returns:
        the value of x unchanged
        
    """
    return x

def add(x: float, y: float) -> float:
    """Adds two numbers
    
    Args:
        x: float 
        y: float

    Returns:
        sum of x and y
        
    """
    return x + y

def neg(x: float) -> float:
    """Negates a number
    
    Args:
        x: float 

    Returns:
        negation of x
        
    """
    return mul(-1, x)

def lt(x: float,y: float) -> bool:
    """Checks if one number is less than another
    
    Args:
        x: float 
        y: float

    Returns:
        result of if x is less than y
        
    """
    return x < y

def eq(x: float, y: float) -> bool:
    """Checks if two numbers are equal
    
    Args:
        x: float 
        y: float

    Returns:
        result of if x is equal to y
        
    """
    return x == y

def max(x: float, y: float) -> float:
    """Returns the larger of two numbers
    
    Args:
        x: float 
        y: float

    Returns:
        value of larger number
        
    """
    if lt(x,y):
        return y
    else:
        return x

def is_close(x: float, y: float) -> bool:
    """Checks if two numbers are close in value
    
    Args:
        x: float 
        y: float

    Returns:
        result of x-y < 1e-25
        
    """
    return lt(
        abs(
            add(x, neg(y))
        ), 1e-25)

def sigmoid(x: float) -> float:
    """Calculates the sigmoid function
    
    Args:
        x: float 

    Returns:
        value of sigmoid(x)      
        
    """
    if x >= 0:
        return 1/(1+exp(neg(x)))
    else:
        return exp(x)/(1+exp(x))

def relu(x: float) -> float:
    """Applies ReLU activation function
    
    Args:
        x: float 

    Returns:
        result of ReLU(x)
        
    """
    if lt(0, x):
        return x
    else:
        return 0.0

def log(x: float) -> float:
    """Calculates natural logarithm
    
    Args:
        x: float 

    Returns:
        result of ln(x)
        
    """
    return math.log(x)

def exp(x: float) -> float:
    """Calculates exponential function
    
    Args:
        x: float 

    Returns:
        result of e^x
        
    """
    return math.exp(x)

def inv(x: float) -> float:
    """Calculates the reciprocal
    
    Args:
        x: float 

    Returns:
        value of 1/x
        
    """
    return 1 / x

def log_back(x: float, y: float) -> float:
    """Computes derivative of log times a second arg
    
    Args:
        x: float 
        y: float

    Returns:
        value of 1/x * y
        
    """
    return mul(y, inv(x))

def inv_back(x: float, y: float) -> float:
    """Computes deriative of reciprocal times a second arg
    
    Args:
        x: float 
        y: float

    Returns:
        -1/x^2 * y
        
    """
    return neg(
        mul(
            inv(mul(x,x)),
            y
        )
    )

def relu_back(x: float, y: float) -> float:
    """Computes derivative of ReLU times a second arg
    
    Args:
        x: float 
        y: float

    Returns:
        y if x > 0 else 0
        
    """
    if lt(0, x):
        return y
    else:
        return 0

    


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(func: Callable[[Any], Any]) -> Callable[[Iterable[Any]], Iterable[Any]]:
    """Higher order function that applies a given function to each element of an iterable
    
    Args:
        func: mapping function 
        ls: iterable of values

    Returns:
        function to transform mapping of iterable
        
        
    """
    def apply(ls: Iterable[Any]) -> Iterable[Any]:

        ret = []
    
        for element in ls:

            ret.append(func(element))
    
        return ret
    
    return apply

def zipWith(func: Callable[[Any, Any], Any]) -> Callable[[Iterable[Any], Iterable[Any]], Iterable[Any]]:
    """Higher-order function that combines elements from two iterables using a given function
    
    Args:
        func: zip function
        x, y: iterable of values

    Returns:
        function to zip x and y

    """
    def apply(x: Iterable[Any], y: Iterable[Any]) -> Iterable[Any]:
        sentinel = object()

        iter_x = iter(x)
        iter_y = iter(y)
        ret = []

        while True:
            element_x = next(iter_x, sentinel)
            element_y = next(iter_y, sentinel)

            if element_x == sentinel or element_y == sentinel:
                return ret
            else:
                ret.append(func(element_x, element_y))

    return apply

def reduce(func: Callable[[Any, Any], Any]) -> Callable[[Iterable[Any]], Any]:
    """Higher order function that reduces an iterable to a single value using a given function

    Args:
        func: reduce function
        ls: iterable of values 
    
    Returns:
        final value of reduced elements
    
    """
    def apply(ls: Iterable[Any]) -> Any:
        it = iter(ls)
        ret = next(it)

        for element in it:
            ret = func(ret, element)

        return ret

    return apply

def negList(iterable: Iterable[float]) -> Iterable[float]:
    """Negate all items in a list using map

    Args:
        iterable: iterable of floats
    
    Returns:
        iterable of negated floats
    
    """
    return map(neg)(iterable)

def addLists(x: Iterable[float], y: Iterable[float]) -> Iterable[float]:
    """Add corresponding elements from two lists using zipWith
    
    Args:
        x: iterable of floats
        y: iterable of floats
    
    Returns:
        iterable of sums of each value
    
    """
    return zipWith(add)(x, y)

def sum(iterable: Iterable[float]) -> float:
    """Sum all elements in a list using reduce
    
    Args:
        iterable: iterable of floats
    
    Returns:
        sum of all values in arr
    
    """
    return reduce(add)(iterable)

def prod(iterable: Iterable[float]) -> float:
    """Calculate product of all elements in a list using reduce
    
    Args:
        iterable: iterable of floats
    
    Returns:
        product of all values in arr
    
    """
    return reduce(mul)(iterable)
