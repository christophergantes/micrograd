from micrograd.engine import Value
import math
from pytest import approx


def test_add():
    a = Value(data=1.0)
    b = Value(data=2.0)

    c = a + b

    assert c.data == (1.0 + 2.0)


def test_mul():
    a = Value(data=3.0)
    b = Value(data=2.0)

    c = a * b

    assert c.data == (3.0 * 2.0)


def test_tanh():
    x = Value(data=0.5)

    y = x.tanh()

    assert y.data == approx(math.tanh(0.5))
