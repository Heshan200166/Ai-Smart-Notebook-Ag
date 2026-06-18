"""
Math Solver Module
====================
Evaluates mathematical expressions extracted from OCR text.
Uses SymPy for safe symbolic computation — no eval().

Supports:
- Basic arithmetic: +, -, *, /, ^, √
- Equations: x^2 - 4 = 0 → x = ±2
- Percentages: 25% of 200
- Square roots: √16, sqrt(16)
"""

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class MathResult:
    """Result of a math evaluation."""
    expression: str          # Cleaned expression
    solution: str            # Computed result
    success: bool            # Whether computation succeeded
    error: Optional[str]     # Error message if failed


class MathSolver:
    """Safe mathematical expression evaluator using SymPy."""

    # Patterns that look like math
    MATH_PATTERNS = [
        r'\d+\s*[\+\-\*\/\^x×÷]\s*\d+',   # 25 + 4, 5 * 8
        r'\d+\s*=',                          # equation
        r'sqrt|√',                           # square root
        r'\d+\s*%',                          # percentage
        r'x\s*[\+\-\*\/\^]',                # variable expressions
    ]

    # Symbol normalization map
    SYMBOL_MAP = {
        '×': '*',
        '÷': '/',
        '^': '**',
        '√': 'sqrt',
        'x': 'x',       # Keep as variable
        'X': 'x',
        '²': '**2',
        '³': '**3',
    }

    def solve(self, text: str) -> MathResult:
        """
        Solve a mathematical expression or equation.

        Args:
            text: Raw text string (possibly from OCR).

        Returns:
            MathResult with the solution or error.
        """
        if not text or not text.strip():
            return MathResult(text, "", False, "Empty expression")

        try:
            import sympy
            from sympy import symbols, solve as sym_solve, sqrt, Rational
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication_application,
                convert_xor
            )
        except ImportError:
            return MathResult(text, "", False, "SymPy not installed")

        # Clean and normalize the expression
        cleaned = self._normalize(text)

        if not cleaned:
            return MathResult(text, "", False, "Could not parse expression")

        try:
            # Check if it's an equation (contains '=')
            if '=' in cleaned:
                return self._solve_equation(cleaned, sympy, sym_solve, symbols)
            else:
                return self._evaluate_expression(cleaned, sympy)

        except Exception as e:
            return MathResult(text, "", False, f"Math error: {str(e)}")

    def _normalize(self, text: str) -> str:
        """
        Normalize OCR text into a valid math expression.

        Handles common OCR misreads and symbol variations.
        """
        result = text.strip()

        # Apply symbol map
        for old, new in self.SYMBOL_MAP.items():
            result = result.replace(old, new)

        # Handle percentage: "25% of 200" → "(25/100)*200"
        pct_match = re.match(r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)', result, re.IGNORECASE)
        if pct_match:
            pct, value = pct_match.groups()
            return f"({pct}/100)*{value}"

        # Simple percentage: "25%" → "25/100"
        pct_simple = re.match(r'^(\d+(?:\.\d+)?)\s*%$', result)
        if pct_simple:
            return f"{pct_simple.group(1)}/100"

        # Handle sqrt notation: "sqrt16" → "sqrt(16)"
        result = re.sub(r'sqrt\s*(\d+)', r'sqrt(\1)', result)

        # Remove non-math characters (keep digits, operators, parentheses, x, ., =)
        result = re.sub(r'[^0-9+\-*/().=x\s]', '', result)

        # Clean up whitespace
        result = result.strip()

        return result

    def _evaluate_expression(self, expr: str, sympy) -> MathResult:
        """Evaluate a mathematical expression (no equation)."""
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        # Check if expression contains variable 'x'
        if 'x' in expr:
            x = sympy.Symbol('x')
            parsed = parse_expr(expr, local_dict={'x': x},
                                transformations=transformations)
            # Simplify the expression
            simplified = sympy.simplify(parsed)
            return MathResult(expr, str(simplified), True, None)
        else:
            parsed = parse_expr(expr, transformations=transformations)
            result = parsed.evalf()

            # Clean up result (remove trailing zeros)
            if result == int(result):
                result_str = str(int(result))
            else:
                result_str = f"{float(result):.6g}"

            return MathResult(expr, result_str, True, None)

    def _solve_equation(self, expr: str, sympy, sym_solve, symbols) -> MathResult:
        """Solve an equation (contains '=')."""
        parts = expr.split('=')
        if len(parts) != 2:
            return MathResult(expr, "", False, "Invalid equation format")

        lhs, rhs = parts[0].strip(), parts[1].strip()

        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )

        x = sympy.Symbol('x')
        local = {'x': x}

        try:
            left = parse_expr(lhs, local_dict=local, transformations=transformations)
            right = parse_expr(rhs if rhs else '0', local_dict=local,
                               transformations=transformations)

            equation = sympy.Eq(left, right)
            solutions = sym_solve(equation, x)

            if not solutions:
                return MathResult(expr, "No solution", True, None)

            sol_strs = [str(s) for s in solutions]
            result = ", ".join(sol_strs)

            return MathResult(expr, f"x = {result}", True, None)

        except Exception as e:
            return MathResult(expr, "", False, f"Could not solve: {str(e)}")

    def is_math_expression(self, text: str) -> bool:
        """
        Heuristic check: does the text look like a math expression?

        Args:
            text: Text string to check.

        Returns:
            True if text appears to be mathematical.
        """
        if not text or len(text.strip()) < 2:
            return False

        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False
