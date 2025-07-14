class Value:
    def __init__(
        self,
        data: float,
        children: tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ):
        self.data: float = data
        self.grad: float = 0.0
        self._prev: set[Value] = set(children)
        self._op: str = _op
        self.label: str = label

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f})"

    def __add__(self, other) -> "Value":
        return Value(data=(self.data + other.data), children=(self, other), _op="+")
