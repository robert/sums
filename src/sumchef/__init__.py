import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, List, Sequence


class Value(ABC):
    """
    Abstract base class representing a value in a constraint satisfaction problem.

    This class serves as the foundation for all value types that can be evaluated
    within a given set of variable bindings.
    """

    @abstractmethod
    def evaluate(self, bindings: dict[str, int]) -> int:
        """
        Evaluate the value given a set of variable bindings.

        Args:
            bindings: Dictionary mapping variable names to their integer values.

        Returns:
            The evaluated integer result.
        """
        pass

    @abstractmethod
    def variables(self) -> List["Variable"]:
        """
        Get all variables used in this value expression.

        Returns:
            List of Variable objects used in this value expression.
        """
        pass


class Variable(Value):
    """
    A Variable is a placeholder for a value that will be determined during
    constraint solving.

    Args:
        name: The identifier for this variable.
    """

    def __init__(self, name: str):
        self.name = name

    def evaluate(self, bindings: dict[str, int]) -> int:
        return bindings[self.name]

    def variables(self) -> List["Variable"]:
        return [self]


class Literal(Value):
    """
    Represents a constant integer.

    Args:
        val: The constant integer value.
    """

    def __init__(self, val: int):
        self.val = val

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.val

    def variables(self) -> List[Variable]:
        return []


@dataclass
class Add(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) + self.operand2.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Subtract(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) - self.operand2.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class Multiply(Value):
    operand1: Value
    operand2: Value

    def evaluate(self, bindings: dict[str, int]) -> int:
        return self.operand1.evaluate(bindings) * self.operand2.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


class Constraint(ABC):
    """
    Abstract base class representing a constraint. Constraints define conditions that must
    be satisfied by the variable bindings.
    """

    @abstractmethod
    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        """
        Check if the constraint is satisfied given variable bindings.

        Args:
            bindings: Dictionary mapping variable names to their values.

        Returns:
            True if the constraint is satisfied, False otherwise.
        """
        pass

    @abstractmethod
    def variables(self) -> List[Variable]:
        """
        Get all variables used in this constraint.

        Returns:
            List of variables used in this constraint.
        """
        pass


class Equal(Constraint):
    """
    Represents an equality constraint between two values.

    Args:
        operand1: First value to compare.
        operand2: Second value to compare.
    """

    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.operand1.evaluate(bindings) == self.operand2.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


@dataclass
class IsLessThan(Constraint):
    """
    Represents a less-than constraint between a value and a threshold.

    Args:
        value: Value to compare.
        threshold: Value that must be greater than the first value.
    """

    value: Value
    threshold: Value

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.value.evaluate(bindings) < self.threshold.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.threshold.variables())


@dataclass
class IsGreaterThan(Constraint):
    """
    Represents a greater-than constraint between a value and a threshold.

    Args:
        value: Value to compare.
        threshold: Value that must be less than the first value.
    """

    value: Value
    threshold: Value

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return self.value.evaluate(bindings) > self.threshold.evaluate(bindings)

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.threshold.variables())


class IsDivisibleBy(Constraint):
    """
    Represents a divisibility constraint between two values.

    Args:
        value: Value to check for divisibility.
        divisible_by: Value that should divide value without remainder.
    """

    def __init__(self, value: Value, divisible_by: Value):
        self.value = value
        self.divisible_by = divisible_by

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return (
            self.value.evaluate(bindings) % self.divisible_by.evaluate(bindings)
        ) == 0

    def variables(self) -> List[Variable]:
        return filter_variables(self.value.variables() + self.divisible_by.variables())


class NOf(Constraint):
    """
    Represents a constraint that exactly N sub-constraints must be satisfied.

    Args:
        sub_constraints: List of constraints to check.
        n: Number of constraints that must be satisfied.
    """

    def __init__(self, sub_constraints: List[Constraint], n: int):
        self.sub_constraints = sub_constraints
        self.n = n

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        return (
            sum(1 for c in self.sub_constraints if c.is_satisfied(bindings)) == self.n
        )

    def variables(self) -> List[Variable]:
        return _flatten([c.variables() for c in self.sub_constraints])


class AdditionCrosses10Boundary(Constraint):
    """
    Constraint that checks if adding two values requires carrying over the ones digit.

    Args:
        operand1: First value to add.
        operand2: Second value to add.
    """

    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 10 + value2 % 10) >= 10

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


class AdditionCrosses100Boundary(Constraint):
    """
    Constraint that checks if adding two values requires carrying over the tens digit.

    Args:
        operand1: First value to add.
        operand2: Second value to add.
    """

    def __init__(self, operand1: Value, operand2: Value):
        self.operand1 = operand1
        self.operand2 = operand2

    def is_satisfied(self, bindings: dict[str, int]) -> bool:
        value1 = self.operand1.evaluate(bindings)
        value2 = self.operand2.evaluate(bindings)
        return (value1 % 100 + value2 % 100) >= 100

    def variables(self) -> List[Variable]:
        return filter_variables(self.operand1.variables() + self.operand2.variables())


def _flatten(lst: list) -> list:
    """
    Recursively flatten a nested list structure.

    Args:
        lst: List that may contain other lists.

    Returns:
        Flattened list with no nested lists.
    """
    flat = []
    for item in lst:
        if isinstance(item, list):
            flat.extend(_flatten(item))
        else:
            flat.append(item)
    return flat


def filter_variables(values: List[Value]) -> List[Variable]:
    """
    Extract unique Variable instances from a list of Values.

    Args:
        values: List of Value objects that may contain Variables.

    Returns:
        List of unique Variable instances found in the input list.
    """
    return list(set([v for v in values if isinstance(v, Variable)]))


def _backtrack(
    variables: List[str], domains: dict[str, List[int]], constraints: List[Constraint]
) -> dict[str, int] | None:
    """
    Internal backtracking solver for constraint satisfaction problems.

    Args:
        variables: List of variable names to assign.
        domains: Dictionary mapping variable names to their possible values.
        constraints: List of constraints that must be satisfied.

    Returns:
        Dictionary mapping variables to values that satisfies all constraints,
        or None if no solution exists.
    """

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
    variables: List[str],
    domains: dict[str, List[int]],
    constraints: List[Constraint],
) -> Generator[dict[str, int], None, None]:
    """
    Generate solutions to a constraint satisfaction problem.

    Args:
        variables: List of variable names to assign.
        domains: Dictionary mapping variable names to their possible values.
        constraints: List of constraints that must be satisfied.

    Yields:
        Dictionary mapping variables to values that satisfies all constraints.
    """
    domains_copy = {var: list(domain) for var, domain in domains.items()}

    while True:
        for domain in domains_copy.values():
            random.shuffle(domain)

        solution = _backtrack(variables, domains_copy, constraints)
        if solution is None:
            break
        yield solution


def find_bindings(
    variables: List[str],
    domains: dict[str, List[int]],
    constraints: List[Constraint],
    n_bindings: int = 1,
) -> List[dict[str, int]]:
    """
    Find multiple solutions to a constraint satisfaction problem.

    Args:
        variables: List of variable names to assign.
        domains: Dictionary mapping variable names to their possible values.
        constraints: List of constraints that must be satisfied.
        n_bindings: Number of solutions to find (default=1).

    Returns:
        List of dictionaries mapping variables to values that satisfy all constraints.
    """
    gen = gen_bindings(variables, domains, constraints)
    all_bindings = []
    for _ in range(n_bindings):
        try:
            all_bindings.append(next(gen))
        except StopIteration:
            break
    return all_bindings


def n_solutions(
    variables: List[str], domains: dict[str, List[int]], constraints: List[Constraint]
) -> int:
    """
    Count the number of solutions to a constraint satisfaction problem.

    Args:
        variables: List of variable names to assign.
        domains: Dictionary mapping variable names to their possible values.
        constraints: List of constraints that must be satisfied.

    Returns:
        Number of unique solutions found.
    """
    return sum(1 for _ in find_bindings(variables, domains, constraints))


def expression_string(
    expression: Value,
    values: dict[str, int],
    hold_out: Variable | None = None,
    underline: Variable | None = None,
) -> str:
    """
    Convert a value expression to a string representation with variable values.

    Args:
        expression: Value expression to convert to string.
        values: Dictionary mapping variable names to their values.
        hold_out: Optional variable to replace with underscores.
        underline: Optional variable to underline in output.

    Returns:
        String representation of the expression with variables replaced by their values.
    """

    def _generate(expr: Value, parent_op: str = None) -> str:
        if isinstance(expr, Variable):
            if expr == hold_out:
                return "_" * len(str(values[expr.name]))
            elif expr == underline:
                return f"<u>{str(values[expr.name])}</u>"
            else:
                return str(values[expr.name])
        elif isinstance(expr, Literal):
            return str(expr.val)
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


def uniform_domains(
    var_names: list[str], domain: Sequence[int]
) -> dict[str, list[int]]:
    return {v: list(domain) for v in var_names}
