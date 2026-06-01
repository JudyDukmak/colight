import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASELINE = r"records\colight-50\anon_4_4_hangzhou_real.json_05_09_21_29_11\train_round"

ENHANCED = r"records\exp-50\anon_4_4_hangzhou_real.json_05_09_06_04_25\train_round"


def baseline_reward(df):

    df["travel_time"] = (
        df["leave_time"] -
        df["enter_time"]
    )

    return -df["travel_time"].mean()


def enhanced_reward(df):

    df["travel_time"] = (
        df["leave_time"] -
        df["enter_time"]
    )

    return (
        -0.25*df["travel_time"].mean()
        -0.25*df["travel_time"].std()
    )


def compute(path, reward_fn):

    results=[]

    rounds=sorted(
        [
            x for x in os.listdir(path)
            if x.startswith("round_")
        ],
        key=lambda x:int(x.split("_")[1])
    )

    for r in rounds:

        gen=os.path.join(
            path,
            r,
            "generator_0"
        )

        if not os.path.exists(gen):
            continue

        csvs=[]

        for f in os.listdir(gen):

            if f.startswith("vehicle_inter"):

                try:
                    csvs.append(
                        pd.read_csv(
                            os.path.join(
                                gen,
                                f
                            )
                        )
                    )
                except:
                    pass

        if len(csvs)==0:
            continue

        df=pd.concat(
            csvs
        )

        reward=reward_fn(df)

        results.append(
            [
                int(r.split("_")[1]),
                reward
            ]
        )

        print(
            r,
            reward
        )

    return pd.DataFrame(
        results,
        columns=[
            "round",
            "reward"
        ]
    )


print("Baseline")

base=compute(
    BASELINE,
    baseline_reward
)

print("\nEnhanced")

enh=compute(
    ENHANCED,
    enhanced_reward
)


base["method"]="Baseline"

enh["method"]="Enhanced"


all_df=pd.concat(
[
base,
enh
]
)


all_df.to_csv(
"reward_comparison.csv",
index=False
)


plt.figure(
figsize=(10,6)
)

plt.plot(
base["round"],
base["reward"],
label="Baseline"
)

plt.plot(
enh["round"],
enh["reward"],
label="Enhanced"
)

plt.xlabel(
"Round"
)

plt.ylabel(
"Reward"
)

plt.title(
"Reward Curve"
)

plt.legend()

plt.grid()

plt.savefig(
"reward_curve.png",
dpi=300
)

plt.show()


print("DONE")