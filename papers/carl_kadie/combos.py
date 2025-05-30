import sympy as sym
from sympy import latex, LambertW, CRootOf, N


# Function to check if an expression leads to a transcendental equation
def try_to_solve_internal(expression, variable):
    
    if isinstance(expression, list):
        expression = expression[0]
        timeout = True
    else:
        timeout = False


    equation = sym.Eq(expression, 0)

    if timeout:
        return (equation, None, "Timeout")

    # Try to solve the equation
    try:
        solutions = sym.solve(equation, variable)
        if solutions:
            if any(sol.has(LambertW) for sol in solutions):
                return (equation, solutions, "Yes, with W, Exp, Log")
            elif any(sol.has(CRootOf) for sol in solutions):
                return (equation, None, "No? (CRootOf)")
            else:
                return (equation, solutions, "Yes, Elementary")
        else:
            return (equation, solutions, "Returns 0 Solutions")
    except Exception as e:
        return (equation, None, f"No? ({e.__class__.__name__})")


def try_to_solve(expression, variable):
    equation, solutions, category = try_to_solve_internal(
        expression, variable
    )
    # if solutions is None, try numeric solving
    if solutions is None:
        try:
            numeric = sym.nsolve(equation, 1)
        except Exception as _:
            try:
                numeric = sym.nsolve(equation, 0)
            except Exception as e:
                numeric = e.__class__.__name__
    elif len(solutions) == 0:
        numeric = sym.nsolve(equation, 0)
    else:
        numeric = N(solutions[0])

    return equation, solutions, category, numeric


def generate_markdown_table_row(equation, solutions, category, numeric):
    # Convert equation and solution to LaTeX format
    equation_md = f"${latex(equation)}$"
    if solutions is None:
        solution_md = ""
    elif len(solutions) == 0:
        solution_md = ""
    elif len(solutions) == 1:
        solution = solutions[0]
        solution_md = f"${latex(solution)}$"
    elif len(solutions) == 2:
        # Show both solutions if there are exactly 2
        solutions_latex = ", ".join([latex(sol) for sol in solutions])
        solution_md = f"$\\{{ {solutions_latex} \\}}$"  # Format as a set
    else:
        op_count = solutions[0].count_ops() + 1
        solution_md = r"\{{ " + latex(solutions[0]) + ", "
        for sol in solutions[1:]:
            op_count += sol.count_ops() + 1
            if op_count >= 30:
                break
            solution_md += f"{latex(sol)}, "
        solution_md += r"\dots \}}"
        solution_md = f"${solution_md}$"

    if numeric is None:
        numeric_md = "*n/a*"
    elif isinstance(numeric, str):
        numeric_md = numeric
    elif numeric.is_real:
        numeric_md = f"{numeric:.4f}"  # Format regular numbers
    else:
        real_part, imag_part = numeric.as_real_imag()
        numeric_md = f"{real_part:.4f} + {imag_part:.4f}i"  # Format complex numbers

    # Create the markdown table row
    markdown_row = f"| {equation_md} | {category} | {solution_md} | {numeric_md} |\n"

    return markdown_row


def generate_markdown_table(tuple_list):
    full_table = "| Equation | Closed-Form (CF) Solution? | CF Solution(s) | A Numeric |\n|----------|----------|----------|-------|\n"
    for equation, solution, category, numeric in tuple_list:
        table_row = generate_markdown_table_row(equation, solution, category, numeric)
        full_table += table_row
    return full_table


def split_on_predicate(lst, predicate):
    result = []
    current_group = []

    for item in lst:
        if predicate(item):
            # If we already have a group, add it to the result
            if current_group:
                result.append(current_group)
            # Start a new group with the item that matched the predicate
            current_group = [item]
        else:
            # If the item does not match the predicate, add it to the current group
            current_group.append(item)

    # Append the last group at the end
    if current_group:
        result.append(current_group)

    return result
