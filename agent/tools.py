"""Agent tools. Tools are built per customer so order lookups are authorized against the caller's identity."""
from __future__ import annotations

import ast
import operator

from langchain_core.tools import StructuredTool

# Synthetic order database.
ORDERS = {
    '1001': {'customer_id': 'C001', 'status': 'shipped', 'carrier': 'UPS', 'eta': '2 business days',
             'email': 'alex@example.com'},
    '1002': {'customer_id': 'C001', 'status': 'delivered', 'delivered': '12 days ago', 'email': 'alex@example.com'},
    '1003': {'customer_id': 'C002', 'status': 'processing', 'email': 'sam@example.com', 'phone': '+1 415-555-0132'},
}

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.USub: operator.neg, ast.Pow: operator.pow,
}


def safe_eval(expression: str) -> float:
    """Evaluate +, -, *, /, ** on numbers only (no names, calls, or attribute access)."""
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f'unsupported expression: {expression!r}')

    return _eval(ast.parse(expression, mode='eval').body)


def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '49.99 * 0.15'."""
    try:
        result = safe_eval(expression)
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f'error: {e}'
    return str(int(result)) if float(result).is_integer() else f'{result:.2f}'


def search_policies(query: str) -> str:
    """Search the refund, shipping, warranty, and account-security policy documents."""
    from rag.vector_store import retrieve  # lazy: needs chromadb + OPENAI_API_KEY

    return '\n\n'.join(f'[{c.source}] {c.text}' for c in retrieve(query))


def lookup_order(order_id: str, customer_id: str) -> str:
    order = ORDERS.get(order_id)
    if order is None:
        return f'Order {order_id} not found.'
    if order['customer_id'] != customer_id:
        return f'Access denied: order {order_id} does not belong to the current customer.'
    return ', '.join(f'{k}={v}' for k, v in order.items() if k != 'customer_id')


def build_tools(customer_id: str, search_fn=search_policies) -> list[StructuredTool]:
    def get_order_status(order_id: str) -> str:
        """Look up the status of one of the current customer's orders by order ID."""
        return lookup_order(order_id, customer_id)

    return [
        StructuredTool.from_function(calculator),
        StructuredTool.from_function(search_fn, name='search_policies', description=search_policies.__doc__),
        StructuredTool.from_function(get_order_status),
    ]
