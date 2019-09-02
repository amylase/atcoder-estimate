import math
from argparse import ArgumentParser

import pandas as pd
import requests
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDRegressor
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize
from scipy.stats import norm


def inverse_adjust_rating(rating, prev_contests):
    if rating == 0:
        return float("nan")
    if rating <= 400:
        rating = 400 * (1 - math.log(400 / rating))
    corr = (math.sqrt(1 - 0.81 ** prev_contests) / (1 - 0.9 ** prev_contests) - 1) / (math.sqrt(19) - 1) * 1200
    return rating + corr


def adjust_difficulty(raw_difficulty):
    if raw_difficulty <= 400:
        return 400 * math.exp(-(400 - raw_difficulty) / 400)
    else:
        return raw_difficulty


def fixed_result_analysis(task_names, df, my_rating=None):
    df = df[df["rating"] > 0]
    for task_screen_name, task_name in task_names.items():
        model = LogisticRegression(solver="lbfgs")
        input = df[["one", "raw_rating"]]
        output = df[task_screen_name + ".ac"]
        model.fit(input, output)
        is_rated_coef = model.coef_[0][0]
        rating_coef = model.coef_[0][1]
        intercept = model.intercept_[0]
        raw_difficulty = -(intercept + is_rated_coef) / rating_coef
        print("{}: {:.2f} ({:.2f}) ({:.6f})".format(task_name, adjust_difficulty(raw_difficulty), raw_difficulty, -rating_coef))

        first_ac = df[task_screen_name + ".elapsed"].min()
        time_model = LinearRegression()
        # time_model = SGDRegressor(loss="huber", penalty="none", epsilon=0.5, tol=None, max_iter=10000)
        # time_df = df[output == 1.]
        time_df = df[(output == 1.) & (df[task_screen_name + ".time"] > first_ac / 2)]
        time_input = time_df[["raw_rating"]]
        time_output = np.log(time_df[task_screen_name + ".time"])
        time_model.fit(time_input, time_output)
        time_coef = time_model.coef_[0]
        time_intercept = time_model.intercept_ if isinstance(time_model.intercept_, float) else time_model.intercept_[0]
        for _t in [25, 50, 100, 200]:
            t = _t * 60 * 10 ** 9
            r = (math.log(t) - time_intercept) / time_coef
            print(f"difficulty {r} (truncate time: {_t} mins)")
        print(f"{time_coef} * rating + {time_intercept}")
        if time_coef > 0:
            print("[warning] slope is positive. ")

        def predict(r):
            return time_model.predict([[r]])[0]

        def exact_time(logtime):
            return math.exp(logtime) / (10 ** 9)

        def predict_exact(r):
            return exact_time(predict(r))

        if my_rating is not None:
            proba = model.predict_proba([[1, my_rating]])[0][1]
            predicted_log_time = predict(my_rating)
            predicted_time = math.exp(predicted_log_time)
            print("predicted solve probability: {:.1f}%, time: {:.2f} [sec]".format(proba * 100, predicted_time / (10 ** 9)))
        raw_rating = time_input['raw_rating']
        time_output_exact = np.exp(time_output) / (10 ** 9)
        plt.scatter(raw_rating, time_output_exact, marker='+')
        # curve
        xs = list(range(-0, 3001, 100))
        means = [predict_exact(r) for r in xs]
        plt.plot(xs, means, color='red')
        # variance
        y_pred = raw_rating.apply(predict)
        residual = time_output - y_pred
        variance = residual.var()
        stddev = math.sqrt(variance)
        lowers = [exact_time(predict(r) - stddev) for r in xs]
        uppers = [exact_time(predict(r) + stddev) for r in xs]
        plt.plot(xs, lowers, color='pink')
        plt.plot(xs, uppers, color='pink')
        # actual smoothed
        cuts = pd.cut(raw_rating, xs)
        sm_xs = raw_rating.groupby(cuts).mean()
        sm_ys = np.exp(time_output.groupby(cuts).mean()) / (10 ** 9)
        plt.plot(sm_xs, sm_ys, color='yellow')
        plt.show()


def running_analysis(task_names, df, current_time):
    # experimental. the output is totally unreliable.
    rated_df = df[df["is_rated"]]
    for task_screen_name, task_name in task_names.items():
        ac_df = rated_df[rated_df[task_screen_name + ".ac"] > 0.5]
        nonac_df = rated_df[rated_df[task_screen_name + ".ac"] < 0.5]

        ac_rating = ac_df["raw_new_rating"].values
        ac_time = ac_df[task_screen_name + ".time"].values

        nonac_rating = nonac_df["raw_new_rating"].values
        nonac_time = (current_time - nonac_df["last_ac"]).values

        diff_scale = 10 ** 6
        def neg_likelihood(param):
            scaled_diff, disc, time_offset, time_slope, time_stddev = param
            diff = scaled_diff * diff_scale

            ac_expected_times = time_slope * ac_rating + time_offset
            ac_solve_probas = 1. / (1. + np.exp(disc * (diff - ac_rating)))
            ac_time_probas = norm.cdf((np.log(ac_time) - ac_expected_times) / time_stddev)
            ac_total_probas = ac_time_probas * ac_solve_probas

            nonac_expected_times = time_slope * nonac_rating + time_offset
            nonac_solve_probas = 1. / (1. + np.exp(disc * (diff - nonac_rating)))
            nonac_time_probas = norm.cdf((np.log(nonac_time) - nonac_expected_times) / time_stddev)
            nonac_total_probas = 1. - nonac_solve_probas * nonac_time_probas

            likelihood = np.sum(np.log(ac_total_probas)) + np.sum(np.log(nonac_total_probas))
            return -likelihood

        res = minimize(
            neg_likelihood,
            np.array([
                0.,
                0.001,
                np.log(200 * 10 ** 9),
                -0.001,
                10.
            ]),
            bounds=[
                (-4000 / diff_scale, 4000 / diff_scale),
                (0.000001, 1.),
                (np.log(10 ** 9), None),
                (None, -0.000001),
                (0.0001, None)
            ],
            method="trust-constr",
            options={
                "maxiter": 1000,
                "disp": False
            }
        )
        print("difficulty: {}, discrimination: {}".format(res.x[0] * diff_scale, res.x[1]))
        print("mean solve time params: {}, {}, variance: ({})".format(res.x[2], res.x[3], res.x[4]))


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
        new_rating = result_row["Rating"]
        prev_contests = result_row["Competitions"]
        user_name = result_row["UserScreenName"]
        # if not is_rated:
        #     continue
        if prev_contests <= 0:
            continue
        if new_rating <= 0:
            continue

        user_row = {
            "one": 1.,
            "is_rated": is_rated,
            "rating": rating,
            "new_rating": new_rating,
            "prev_contests" : prev_contests,
            "raw_rating": inverse_adjust_rating(rating, prev_contests),
            "raw_new_rating": inverse_adjust_rating(new_rating, prev_contests),
            "user_name": user_name
        }
        for task_name in task_names:
            user_row[task_name + ".ac"] = 0.
            user_row[task_name + ".time"] = -1.
            user_row[task_name + ".elapsed"] = 10 ** 200

        prev_accepted_times = [0] + [task_result["Elapsed"] for task_result in result_row["TaskResults"].values() if task_result["Score"] > 0]
        user_row["last_ac"] = max(prev_accepted_times)
        for task_screen_name, task_result in result_row["TaskResults"].items():
            if task_result["Score"] > 0:
                user_row[task_screen_name + ".ac"] = 1.
                elapsed = task_result["Elapsed"]
                penalty = task_result["Penalty"] * 5 * 60 * (10 ** 9)
                user_row[task_screen_name + ".elapsed"] = elapsed
                user_row[task_screen_name + ".time"] = penalty + elapsed - max(t for t in prev_accepted_times if t < elapsed)
        user_results.append(user_row)
    df = pd.DataFrame(data=user_results)

    if results["Fixed"]:
        fixed_result_analysis(task_names, df, 2360)
    else:
        now = datetime.now()
        start_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
        elapsed = (now - start_time).seconds * (10 ** 9)
        running_analysis(task_names, df, elapsed)

def main():
    parser = ArgumentParser()
    parser.add_argument("--contest", default="abc138")
    args = parser.parse_args()
    estimate_contest_difficulty(args.contest)


if __name__ == '__main__':
    main()