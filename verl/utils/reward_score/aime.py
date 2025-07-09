import re
import random
import ast
import operator


def extract_solution(solution_str):
    solution_str = solution_str.split("after \"Answer:\".")[1].strip().split("\n")[-1]
    solution = re.search("Answer:\\s*(-?[0-9]+)", solution_str)
    if solution is None:
        final_answer = None
    else:
        solution = solution.group(0)
        final_answer = solution.split('Answer:')[1].strip("$").strip()
    return solution_str, solution, final_answer

def compute_score(solution_str, ground_truth, error_score=0.0, format_score=0.1, score=1.):
    solution = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Solution_str: {solution[0]}")
        print(f"Extracted solution: {solution[1]}")
        print(f"Final answer: {solution[2]}")
        print(f"Ground_truth: {ground_truth}")
    if solution is None:
        if do_print:
            print(f"No solution found.")
        return error_score
    
    # Validate equation uses correct numbers
    if solution[2]!=ground_truth:
        if do_print:
            print(f"Wrong answer.")
        return format_score
    else:
        if do_print:
            print(f"Correct answer.")
        return score
