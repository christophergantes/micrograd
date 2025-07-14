from micrograd.engine import Value


def test_add():
    a = Value(data=1.0, label="a")
    b = Value(data=2.0, label="b")

    c = a + b

    assert c.data == 3.0
