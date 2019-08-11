import math
from argparse import ArgumentParser

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression


def inverse_adjust_rating(rating, prev_contests):
    if rating <= 400:
        rating = 400 * (1 - math.log(400 / rating))
    corr = (math.sqrt(1 - 0.81 ** prev_contests) / (1 - 0.9 ** prev_contests) - 1) / (math.sqrt(19) - 1) * 1200
    return rating + corr


def adjust_difficulty(raw_difficulty):
    if raw_difficulty <= 400:
        return 400 * math.exp(-(400 - raw_difficulty) / 400)
    else:
        return raw_difficulty


def estimate_contest_difficulty(contest_name: str):
    results = requests.get("https://atcoder.jp/contests/{}/standings/json".format(contest_name)).json()
    task_names = {task["TaskScreenName"]: task["TaskName"] for task in results["TaskInfo"]}

    user_results = []
    for result_row in results["StandingsData"]:
        total_submissions = result_row["TotalResult"]["Count"]
        if total_submissions == 0:
            continue

        is_rated = result_row["IsRated"]
        rating = result_row["OldRating"]
        prev_contests = result_row["Competitions"]
        user_name = result_row["UserScreenName"]
        # if not is_rated:
        #     continue
        if rating == 0:
            continue

        user_row = {
            "is_rated": is_rated,
            "rating": rating,
            "prev_contests" : prev_contests,
            "raw_rating": inverse_adjust_rating(rating, prev_contests),
            "user_name": user_name
        }
        for task_name in task_names:
            user_row[task_name] = 0.

        for task_screen_name, task_result in result_row["TaskResults"].items():
            if task_result["Score"] > 0:
                user_row[task_screen_name] = 1.
        user_results.append(user_row)
    df = pd.DataFrame(data=user_results)

    for task_screen_name, task_name in task_names.items():
        model = LogisticRegression(solver="lbfgs")
        input = df[["is_rated", "raw_rating"]]
        output = df[task_screen_name]
        model.fit(input, output)
        is_rated_coef = model.coef_[0][0]
        rating_coef = model.coef_[0][1]
        intercept = model.intercept_[0]
        raw_difficulty = -(intercept + is_rated_coef) / rating_coef
        print("{}: {:.2f}".format(task_name, adjust_difficulty(raw_difficulty)))


def main():
    parser = ArgumentParser()
    parser.add_argument("--contest", default="abc135")
    args = parser.parse_args()
    estimate_contest_difficulty(args.contest)


if __name__ == '__main__':
    main()