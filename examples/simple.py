import random

from sumchef import (
    Equal,
    IsDivisibleBy,
    IsGreaterThan,
    IsLessThan,
    Literal,
    Multiply,
    Variable,
    expression_string,
    find_bindings,
)

"""
Prints a list of sums of the form XY0 * Z = ABCD.
"""

x = Variable("x")
y = Variable("y")
z = Variable("z")
lhs = Multiply(x, y)
rhs = z

constraints = [
    IsGreaterThan(x, Literal(50)),
    IsLessThan(x, Literal(1000)),
    IsLessThan(y, Literal(20)),
    IsDivisibleBy(x, Literal(10)),
    Equal(lhs, rhs),
]
domains = {v.name: list(range(1, 10000)) for v in [x, y, z]}
bindings = find_bindings(["x", "y", "z"], domains, constraints, 10)

for b in bindings:
    hold_out = random.choice([x, y, z])
    lhs_expr = expression_string(lhs, b, hold_out=hold_out)
    rhs_expr = expression_string(rhs, b, hold_out=hold_out)
    print(f"{lhs_expr} = {rhs_expr}")
