import math
from typing import List, Set
import requests
import time


def safe_sigmoid(x):
    return 1. / (1. + math.exp(min(-x, 750)))


def iif(theta: float, alpha: float, difficulty: float) -> float:
    p = safe_sigmoid(alpha * (theta - difficulty))
    return (alpha ** 2) * p * (1. - p)


def inverse_adjust_rating(rating, prev_contests):
    if rating <= 0:
        return float("nan")
    if rating <= 400:
        rating = 400 * (1 - math.log(400 / rating))
    adjustment = (math.sqrt(1 - (0.9 ** (2 * prev_contests))) /
                  (1 - 0.9 ** prev_contests) - 1) / (math.sqrt(19) - 1) * 1200
    return rating + adjustment


def fetch_problems():
    time.sleep(0.5)
    problem_models = requests.get("https://kenkoooo.com/atcoder/resources/problem-models.json").json()
    return {problem_id: model for problem_id, model in problem_models.items() if "difficulty" in model}


def fetch_inner_rating(user: str):
    history = requests.get(f"https://atcoder.jp/users/{user}/history/json").json()
    rated_count = len([participation for participation in history if participation["IsRated"]])
    external_rating = history[-1]["NewRating"]
    return inverse_adjust_rating(external_rating, rated_count)


def fetch_submissions(user: str):
    time.sleep(0.5)
    return requests.get(f"https://kenkoooo.com/atcoder/atcoder-api/results?user={user}").json()


def generate_contest(participants: List[str], n_problems: int, adjustment: float=0.) -> Set[str]:
    problems = fetch_problems()
    candidate_problems = problems.keys()
    user_ratings = []
    for user in participants:
        solved_problems = {submission["problem_id"] for submission in fetch_submissions(user) if submission["result"] == "AC"}
        candidate_problems -= solved_problems
        user_ratings.append(fetch_inner_rating(user) + adjustment)
    print(f"Select from {len(candidate_problems)} problems.")

    info_vectors = []
    for candidate in candidate_problems:
        model = problems[candidate]
        info_vector = []
        for rating in user_ratings:
            info_vector.append(iif(rating, model["discrimination"], model["difficulty"]))
        info_vectors.append((info_vector, candidate))

    recommendations = set()
    current_vector = [0.] * len(participants)
    for _ in range(min(n_problems, len(candidate_problems))):
        next_score, next_problem, next_vector = -1, None, []
        for vector, candidate in info_vectors:
            if candidate in recommendations:
                continue
            candidate_vector = [current_vector[i] + vector[i] for i in range(len(participants))]
            candidate_score = min(candidate_vector)
            if candidate_score > next_score:
                next_score, next_problem, next_vector = candidate_score, candidate, candidate_vector
        current_vector = next_vector
        recommendations.add(next_problem)

    for reco in recommendations:
        print(f"{reco}: {int(problems[reco]['difficulty'])}")
    return recommendations


if __name__ == '__main__':
    generate_contest(["amylase", "kenkoooo"], 4)