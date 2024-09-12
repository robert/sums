import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generator, List


class Value(ABC):
    @abstractmethod
    def evaluate(self, bindings: Dict[str, int]) -> int:
        pass

    @abstractmethod
    def variables(self):
        pass


class Variable(Value):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return bindings[self.name]

    def variables(self):
        return [self]


class Literal(Value):
    def __init__(self, val: int):
        self.val = val

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return self.val

    def variables(self):
        return []


@dataclass
class Add(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) + self.operand2.evaluate(bindings)

    def variables(self):
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Subtract(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) - self.operand2.evaluate(bindings)

    def variables(self):
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Multiply(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: Dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) * self.operand2.evaluate(bindings)

    def variables(self):
        return filter_variables(self.operand1.variables() + self.operand2.variables())


class Constraint(ABC):
    @abstractmethod
    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        pass

    @abstractmethod
    def variables(self) -> List[str]:
        pass


class Equal(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        return self.operand1.evaluate(bindings) == self.operand2.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class IsLessThan(Constraint):
    value: Value
    threshold: Value

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        return self.value.evaluate(bindings) < self.threshold.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.threshold.variables())


@dataclass
class IsGreaterThan(Constraint):
    value: Value
    threshold: Value

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        return self.value.evaluate(bindings) > self.threshold.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.threshold.variables())


class IsDivisibleBy(Constraint):
    def __init__(self, value: Value, divisible_by: Value):
        self.value = value
        self.divisible_by = divisible_by

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        return (
            self.value.evaluate(bindings) % self.divisible_by.evaluate(bindings)
        ) == 0

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.divisible_by.variables())


class NOf(Constraint):
    def __init__(self, sub_constraints: list[Constraint], n: int):
        self.sub_constraints = sub_constraints
        self.n = n

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        return (
            sum(1 for c in self.sub_constraints if c.is_satisfied(bindings)) == self.n
        )

    def variables(self) -> List[Variable]:
        return flatten([c.variables() for c in self.sub_constraints])


class AdditionCrosses10Boundary(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 10 + value2 % 10) >= 10

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


class AdditionCrosses100Boundary(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: Dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 100 + value2 % 100) >= 100

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


def flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def filter_variables(values: list[Value]) -> List[Variable]:
    return list(set([v for v in values if isinstance(v, Variable)]))


def backtrack(
    variables: List[str], domains: Dict[str, List[int]], constraints: List[Constraint]
) -> Generator[Dict[str, int], None, None]:
    def is_consistent(assignment: Dict[str, int]) -> bool:
        return all(
            constraint.is_satisfied(assignment)
            for constraint in constraints
            if set([v.name for v in constraint.variables()]).issubset(
                set(assignment.keys())
            )
        )

    def backtrack_helper(
        assignment: Dict[str, int],
    ) -> Generator[Dict[str, int], None, None]:
        if len(assignment) == len(variables):
            if is_consistent(assignment) and not all(
                v == 0 for v in assignment.values()
            ):
                print(f"{assignment=}")
                yield assignment.copy()
        else:
            var = next(v for v in variables if v not in assignment)
            for value in domains[var]:
                assignment[var] = value
                if is_consistent(assignment):
                    yield from backtrack_helper(assignment)
                del assignment[var]

    yield from backtrack_helper({})


def find_bindings(
    variables: List[str], domains: Dict[str, List[int]], constraints: List[Constraint]
) -> Generator[Dict[str, int], None, None]:
    backtrack_gen = backtrack(variables, domains, constraints)
    while True:
        try:
            for d in domains.values():
                random.shuffle(d)
            yield next(backtrack_gen)
            # Reset the generator after each successful binding
            backtrack_gen = backtrack(variables, domains, constraints)
        except StopIteration:
            break


def print_grid(solution: Dict[str, int]):
    grid = [
        ["a", "+", "b", "=", "c"],
        ["+", " ", "+", " ", "+"],
        ["d", "+", "e", "=", "f"],
        ["=", " ", "=", " ", "="],
        ["g", "+", "h", "=", "i"],
    ]

    max_width = max(len(str(val)) for val in solution.values())

    for row in grid:
        formatted_row = []
        for cell in row:
            if cell in solution:
                formatted_row.append(str(solution[cell]).center(max_width))
            else:
                formatted_row.append(cell.center(max_width))
        print(" ".join(formatted_row))
    print()  # Empty line after each grid


def expression_string(
    expression: Value,
    values: dict[str, int],
    hold_out: Variable | None = None,
    underline: Variable | None = None,
) -> str:
    def _generate(expr: Value, parent_op: str = None) -> str:
        print(f"{expr=}")
        print(f"{underline=}")
        if isinstance(expr, Variable):
            if expr == hold_out:
                return "_" * len(str(values[expr.name]))
            elif expr == underline:
                print("ANKNSJA")
                return f"<u>{str(values[expr.name])}</u>"
            else:
                return str(values[expr.name])
        elif isinstance(expr, Literal):
            return str(expr.value)
        elif isinstance(expr, Add):
            left = _generate(expr.operand1, "+")
            right = _generate(expr.operand2, "+")
            result = f"{left} + {right}"
            if parent_op in ("*", "-"):
                return f"({result})"
            return result
        elif isinstance(expr, Subtract):
            left = _generate(expr.operand1, "-")
            right = _generate(expr.operand2, "-")
            result = f"{left} - {right}"
            if parent_op in ("*", "-"):
                return f"({result})"
            return result
        elif isinstance(expr, Multiply):
            left = _generate(expr.operand1, "*")
            right = _generate(expr.operand2, "*")
            return f"{left} * {right}"
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    return _generate(expression)


if __name__ == "__main__":
    # Example usage
    a, b, c, d, e, f, g, h, i = [Variable(var) for var in "abcdefghi"]

    constraints = [
        Equal(Add(a, b), c),
        Equal(Add(d, e), f),
        Equal(Add(g, h), i),
        Equal(Add(a, d), g),
        Equal(Add(b, e), h),
        Equal(Add(c, f), i),
    ]

    vars = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    domains = {v: list(range(1, 51)) for v in vars}

    count = 0
    for solution in find_bindings(vars, domains, constraints):
        print_grid(solution)
        # count += 1
        # if count >= 5:  # Limit to 5 solutions for demonstration
        #    break

    print(f"Total solutions found: {count}")
