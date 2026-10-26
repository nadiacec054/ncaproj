
import ast
import math
import operator
import re


BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

COMPARISON_OPERATORS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.Gt: operator.gt,
    ast.LtE: operator.le,
    ast.GtE: operator.ge,
}


def evaluate_node(node):
    if isinstance(node, ast.Expression):
        return evaluate_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.Name):
        if node.id == "pi":
            return math.pi
        raise ValueError(f"Unknown symbol: {node.id}")

    if isinstance(node, ast.BinOp):
        operation = BINARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported arithmetic operator.")

        left = evaluate_node(node.left)
        right = evaluate_node(node.right)
        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operation = UNARY_OPERATORS.get(type(node.op))
        if operation is None:
            raise ValueError("Unsupported unary operator.")
        return operation(evaluate_node(node.operand))

    if isinstance(node, ast.Compare):
        left = evaluate_node(node.left)

        for op, comparator in zip(
            node.ops, node.comparators
        ):
            operation = COMPARISON_OPERATORS.get(type(op))
            if operation is None:
                raise ValueError("Unsupported comparison.")

            right = evaluate_node(comparator)

            if not operation(left, right):
                return False

            left = right

        return True

    raise ValueError("Invalid mathematical expression.")


def solve_expression(expression):
    expression = expression.strip()

    if not expression:
        raise ValueError("Enter an expression first.")

    # Convert common mathematical notation to Python notation.
    expression = expression.replace("×", "*")
    expression = expression.replace("÷", "/")
    expression = expression.replace("π", "pi")
    expression = expression.replace("−", "-")
    expression = expression.replace("≠", "!=")

    # Treat a standalone equals sign as an equality comparison.
    expression = re.sub(
        r"(?<![<>=!])=(?!=)",
        "==",
        expression
    )

    try:
        tree = ast.parse(expression, mode="eval")
        return evaluate_node(tree)
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero.")
    except (SyntaxError, OverflowError) as error:
        raise ValueError("Please enter a valid expression.") from error


if __name__ == "__main__":
    while True:
        expression = input(
            "Enter an expression (or 'quit' to exit): "
        )

        if expression.strip().lower() == "quit":
            break

        try:
            print("Answer:", solve_expression(expression))
        except (ValueError, TypeError) as error:
            print("Error:", error)
