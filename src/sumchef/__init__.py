import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator


class Value(ABC):
    @abstractmethod
    def evaluate(self, bindings: dict[str, int]) -> int:
        pass

    @abstractmethod
    def variables(self):
        pass


class Variable(Value):
    def __init__(self, name: str):
        self.name = name

    def evaluate(self, bindings: dict[str, int]) -> int:
        return bindings[self.name]

    def variables(self):
        return [self]


class Literal(Value):
    def __init__(self, val: int):
        self.val = val

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.val

    def variables(self):
        return []


@dataclass
class Add(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) + self.operand2.evaluate(bindings)

    def variables(self):
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Subtract(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) - self.operand2.evaluate(bindings)

    def variables(self):
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Multiply(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) * self.operand2.evaluate(bindings)

    def variables(self):
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


class Constraint(ABC):
    @abstractmethod
    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        pass

    @abstractmethod
    def variables(self) -> list[str]:
        pass


class Equal(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.operand1.evaluate(bindings) == self.operand2.evaluate(bindings)

    def variables(self) -> list[Variable]:
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class IsLessThan(Constraint):
    value: Value
    threshold: Value

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.value.evaluate(bindings) < self.threshold.evaluate(bindings)

    def variables(self) -> list[Variable]:
        return _filter_variables(self.value.variables() + self.threshold.variables())


@dataclass
class IsGreaterThan(Constraint):
    value: Value
    threshold: Value

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.value.evaluate(bindings) > self.threshold.evaluate(bindings)

    def variables(self) -> list[Variable]:
        return _filter_variables(self.value.variables() + self.threshold.variables())


class IsDivisibleBy(Constraint):
    def __init__(self, value: Value, divisible_by: Value):
        self.value = value
        self.divisible_by = divisible_by

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return (
            self.value.evaluate(bindings) % self.divisible_by.evaluate(bindings)
        ) == 0

    def variables(self) -> list[Variable]:
        return _filter_variables(self.value.variables() + self.divisible_by.variables())


class NOf(Constraint):
    def __init__(self, sub_constraints: list[Constraint], n: int):
        self.sub_constraints = sub_constraints
        self.n = n

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return (
            sum(1 for c in self.sub_constraints if c.is_satisfied(bindings)) == self.n
        )

    def variables(self) -> list[Variable]:
        return _flatten([c.variables() for c in self.sub_constraints])


# TODO: AdditionCrossesBoundary
class AdditionCrosses10Boundary(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 10 + value2 % 10) >= 10

    def variables(self) -> list[Variable]:
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


class AdditionCrosses100Boundary(Constraint):
    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 100 + value2 % 100) >= 100

    def variables(self) -> list[Variable]:
        return _filter_variables(self.operand1.variables() + self.operand2.variables())


def _flatten(lst: list):
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


def _filter_variables(values: list[Value]) -> list[Variable]:
    return list(set([v for v in values if isinstance(v, Variable)]))


def _backtrack(
    variables: list[str], domains: dict[str, list[int]], constraints: list[Constraint]
) -> dict[str, int] | None:
    def is_consistent(assignment: dict[str, int]) -> bool:
        return all(
            constraint.is_satisfied(assignment)
            for constraint in constraints
            if set([v.name for v in constraint.variables()]).issubset(
                set(assignment.keys())
            )
        )

    def backtrack_helper(
        assignment: dict[str, int],
    ) -> dict[str, int] | None:
        if len(assignment) == len(variables):
            if is_consistent(assignment) and not all(
                v == 0 for v in assignment.values()
            ):
                return assignment.copy()
            return None

        var = next(v for v in variables if v not in assignment)
        for value in domains[var]:
            assignment[var] = value
            if is_consistent(assignment):
                result = backtrack_helper(assignment)
                if result is not None:
                    return result
            del assignment[var]
        return None

    return backtrack_helper({})


def gen_bindings(
    variables: list[str],
    domains: dict[str, list[int]],
    constraints: list[Constraint],
) -> Generator[dict[str, int], None, None]:
    domains_copy = {var: list(domain) for var, domain in domains.items()}

    while True:
        for domain in domains_copy.values():
            random.shuffle(domain)

        solution = _backtrack(variables, domains_copy, constraints)
        if solution is None:
            break
        yield solution


def find_bindings(
    variables: list[str],
    domains: dict[str, list[int]],
    constraints: list[Constraint],
    n_bindings: int = 1,
) -> list[dict[str, int]]:
    gen = gen_bindings(variables, domains, constraints)
    all_bindings = []
    for _ in range(n_bindings):
        try:
            all_bindings.append(next(gen))
        except StopIteration:
            break
    return all_bindings


def n_solutions(
    variables: list[str], domains: dict[str, list[int]], constraints: list[Constraint]
):
    return sum(1 for _ in find_bindings(variables, domains, constraints))


def expression_string(
    expression: Value,
    values: dict[str, int],
    hold_out: Variable | None = None,
    underline: Variable | None = None,
) -> str:
    def _generate(expr: Value, parent_op: str = None) -> str:
        if isinstance(expr, Variable):
            if expr == hold_out:
                return "_" * len(str(values[expr.name]))
            elif expr == underline:
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
