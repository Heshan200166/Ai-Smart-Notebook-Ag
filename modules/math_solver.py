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
- OCR-friendly: tolerates misreads like 'O' → '0', 'l' → '1'
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
        r'\d+\s*[xX]\s*\d+',                # multiplication with x
    ]

    # OCR misread corrections (applied before math parsing)
    OCR_CORRECTIONS = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        'g': '9', 'q': '9',
        'A': '4',
        'T': '7',
        't': '+',          # '+' is very commonly misread as 't'
        'f': '+',          # '+' is sometimes misread as 'f'
        'y': 'x',          # Treat 'y' as 'x' to allow solving single-variable equations in y
        'Y': 'x',
        '×': '*', '✕': '*', 'X': '*',
        '÷': '/',
        '−': '-', '–': '-', '—': '-',
        '^': '**',
        '√': 'sqrt',
        '²': '**2',
        '³': '**3',
        '{': '(', '}': ')',
        '[': '(', ']': ')',
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

        Very forgiving — handles common OCR misreads, stray characters,
        and multiple notation styles for operators.
        """
        result = text.strip()

        # Handle common OCR misreads for equals sign: "==" or "::" or ":" or "::="
        result = re.sub(r'==|::|::=', '=', result)
        # If there's no '=' but we have a ':' or '::', replace it with '=' if surrounded by math terms
        if '=' not in result:
            result = re.sub(r'\s*:\s*(?=\d|x|y|a|b)', ' = ', result)
            if '=' not in result:
                result = re.sub(r':+', '=', result)

        # Handle percentage: "25% of 200" → "(25/100)*200"
        pct_match = re.match(r'(\d+(?:\.\d+)?)\s*%\s*(?:of\s+)?(\d+(?:\.\d+)?)', result, re.IGNORECASE)
        if pct_match:
            pct, value = pct_match.groups()
            return f"({pct}/100)*{value}"

        # Simple percentage: "25%" → "25/100"
        pct_simple = re.match(r'^(\d+(?:\.\d+)?)\s*%$', result)
        if pct_simple:
            return f"{pct_simple.group(1)}/100"

        # Step 1: Detect if 'x' is being used as a variable (in equations)
        has_equation = '=' in result
        has_variable_x = bool(re.search(r'x\s*[\*\*\^²]|x\s*[+\-]|[+\-]\s*x|\bx\b\s*=', result, re.IGNORECASE))

        # Step 2: Extract and protect math function names FIRST
        # Find all function-like words and mark their positions
        func_names = ['sqrt', 'sin', 'cos', 'tan', 'log', 'ln', 'abs']
        found_funcs = []
        lower_result = result.lower()
        for fn in func_names:
            idx = lower_result.find(fn)
            while idx != -1:
                found_funcs.append((idx, idx + len(fn), fn))
                idx = lower_result.find(fn, idx + 1)

        # Build character list, protecting function characters
        protected_indices = set()
        for start, end, _ in found_funcs:
            for i in range(start, end):
                protected_indices.add(i)

        # Step 3: Apply OCR corrections character by character
        # BUT skip characters that are part of function names
        normalized = []
        for i, ch in enumerate(result):
            if i in protected_indices:
                normalized.append(ch.lower())  # Keep function chars as-is (lowercase)
            elif ch in self.OCR_CORRECTIONS:
                if ch.lower() == 'x' and (has_equation or has_variable_x):
                    normalized.append('x')
                else:
                    normalized.append(self.OCR_CORRECTIONS[ch])
            else:
                normalized.append(ch)
        result = ''.join(normalized)

        # Handle sqrt notation: "sqrt16" → "sqrt(16)"
        result = re.sub(r'sqrt\s*(\d+)', r'sqrt(\1)', result)

        # Step 4: Extract just math characters
        # Keep: digits, operators, parens, x, dots, equals, spaces, AND function letters
        result = re.sub(r'[^0-9a-z+\-*/().=\s]', '', result)

        # Step 5: Fix common OCR spacing issues
        # Add * between number and opening paren: "2(3)" → "2*(3)"
        result = re.sub(r'(\d)\s*\(', r'\1*(', result)
        # Add * between closing paren and number: "(3)2" → "(3)*2"
        result = re.sub(r'\)\s*(\d)', r')*\1', result)

        # Step 5: Clean up multiple spaces and operators
        result = re.sub(r'\s+', ' ', result).strip()
        # Remove trailing operators
        result = re.sub(r'[+\-*/]+$', '', result).strip()
        # Remove leading operators (except minus for negative)
        result = re.sub(r'^[+*/]+', '', result).strip()

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

        # Check if it has at least one digit and one operator
        has_digit = bool(re.search(r'\d', text))
        has_operator = bool(re.search(r'[+\-*/=^×÷%]', text))

        if has_digit and has_operator:
            return True

        for pattern in self.MATH_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False
