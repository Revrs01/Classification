from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import pandas as pd

# Define the dataset size
dataset_size = 30000


def plot_diagram(data, name):
    plt.hist(data)
    plt.title(name)
    plt.savefig(f'./figure/with noise/{name}.png')
    plt.clf()


if __name__ == '__main__':
# =========================================================================================
    # generate Gender
    _, gender = make_classification(
        n_samples=dataset_size,
        n_classes=2,
        weights=[0.5, 0.5],
        random_state=39,
    )
    # gender = ["man" if x else "woman" for x in gender]
    gender = gender.tolist()
    random.shuffle(gender)

# =========================================================================================
    # generate occupation
    _, occupation = make_classification(
        n_samples=dataset_size,
        n_classes=2,
        weights=[0.25, 0.75],
        random_state=93
    )

    # occupation = ["student" if not x else "worker" for x in occupation]
    occupation = occupation.tolist()
    random.shuffle(occupation)

# =========================================================================================
    # generate educational qualification by giving corresponding weights
    educational_qualification = []

    for i in range(dataset_size):
        educational_qualification.append(random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.1, 0.2, 0.5, 0.1])[0])
    random.shuffle(educational_qualification)

# =========================================================================================
    # generate income using normal distribution
    standard_deviation = 50000  # assumed value
    mean_income = 50000

    income = np.random.normal(mean_income, standard_deviation, dataset_size)
    income = income.astype(int)
    income = income.clip(28000, 200000)
    income = income.tolist()
    random.shuffle(income)

# =========================================================================================
    # generate number of installment
    installment = []
    number_of_installment = [0, 3, 6, 9, 12, 18, 24, 36, 48]
    installment_weight = [0.3, 0.2, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05]

    for i in range(dataset_size):
        installment.append(random.choices(number_of_installment, weights=installment_weight)[0])

    random.shuffle(installment)

# =========================================================================================
    # generate number of arrears
    arrears = []

    for i in range(dataset_size):
        arrears.append(random.choices([0, 1, 2], weights=[0.85, 0.13, 0.02])[0])

    random.shuffle(arrears)

# =========================================================================================
    # generate deposit
    deposit = np.random.normal(70000, 70000, dataset_size)
    deposit = np.clip(deposit, 30000, 300000)
    deposit = deposit.astype(int)
    deposit = deposit.tolist()
    random.shuffle(deposit)

# =========================================================================================
    # generate shopping frequency
    shopping_frequency = np.random.normal(5, 4, dataset_size)
    shopping_frequency = np.clip(shopping_frequency, 0, 10)
    shopping_frequency = shopping_frequency.astype(int)
    shopping_frequency = shopping_frequency.tolist()
    random.shuffle(shopping_frequency)

# =========================================================================================
    # generate average cost per shopping
    average_cost = np.random.normal(25000, 30000, dataset_size)
    average_cost = np.clip(average_cost, 3000, 100000)
    average_cost = average_cost.astype(int)

    average_cost = average_cost.tolist()
    random.shuffle(average_cost)

# =========================================================================================
    # generate age
    ages = []

    for i in range(round(dataset_size * 0.25)):
        ages.append(random.choice([j for j in range(20, 29)]))

    worker_age = np.random.normal(35, 5, round(dataset_size * 0.75))

    ages.extend(worker_age.tolist())
    ages = np.array(ages)
    ages = np.clip(ages, 20, 50)
    ages = ages.astype(int)
    ages = ages.tolist()
    random.shuffle(ages)

# =========================================================================================
    # generate number of browsing
    number_of_browsing = []
    for i in range(dataset_size):
        number_of_browsing.append(
            random.choices([j for j in range(1, 11)], weights=[0.1, 0.15, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, ])[
                0])

    random.shuffle(number_of_browsing)

# =========================================================================================
    # generate number of people to raise
    number_of_raise = np.random.normal(1, 1, dataset_size)
    number_of_raise = np.clip(number_of_raise, 0, 4)
    number_of_raise = number_of_raise.astype(int)
    number_of_raise = number_of_raise.tolist()
    random.shuffle(number_of_raise)

# =========================================================================================
    # generate passive income
    # Set the parameters for the exponential distribution
    scale_parameter = 8000  # Adjust to control the skewness

    # Generate 10,000 random integers
    passive_income = np.random.exponential(scale=scale_parameter, size=dataset_size).astype(int)

    # Ensure that the generated integers are within the specified range (0 to 40,000)
    passive_income = np.clip(passive_income, 0, 40000)

    # Shuffle the generated integers to make them more random
    np.random.shuffle(passive_income)
    passive_income = passive_income.tolist()
    random.shuffle(passive_income)

    # pack to json

    dataset = []

    for index, items in enumerate(
            zip(gender, occupation, educational_qualification, income, installment, arrears, deposit,
                shopping_frequency,
                average_cost, ages, number_of_browsing, number_of_raise, passive_income)):
        dataset.append({
            "index": index,
            "gender": items[0],
            "occupation": items[1],
            "educational_qualification": items[2],
            "income": items[3],
            "installment": items[4],
            "arrears": items[5],
            "deposit": items[6],
            "shopping_frequency": items[7],
            "average_cost": items[8],
            "ages": items[9],
            "number_of_browsing": items[10],
            "number_of_raise": items[11],
            "passive_income": items[12],
        })

# =========================================================================================
    # generate noise data
    #
    # for i in dataset:
    #     if i["occupation"] == 0:
    #         i["shopping_frequency"] = int(i["shopping_frequency"] * 1.3)
    #         i["income"] = random.choice([x for x in range(8000, 15001, 1000)])
    #         i["number_of_raise"] = random.choices([0, 1, 2], weights=[0.9, 0.07, 0.03])[0]
    #         if i["installment"] != 0:
    #             i["installment"] = random.choice([6, 9, 12, 18, 24, 36, 48])
    #
    #     if i["gender"] == 0:
    #         i["number_of_browsing"] = int(i["number_of_browsing"] * 1.8)
    #
    #     if i["average_cost"] >= 40000:
    #         i["installment"] = random.choice([6, 9, 12, 18, 24, 36, 48])
    #
    #     if i["educational_qualification"] >= 4:
    #         i["income"] = int(i["income"] * 1.15)
    #
    #     if i["number_of_browsing"] >= 5:
    #         i["shopping_frequency"] = int(i["shopping_frequency"] * 1.2)
    #
    #

# =========================================================================================
    # save dataset to js file
    # run the program 3 times to generate all 3 type of dataset
    # remove one write file script to generate correspond dataset
    #
    # with open('./inputs/dataset_train.js', 'w') as js_file:
    #     json.dump(dataset, js_file, indent=13)

    # with open('./inputs/dataset_test.js', 'w') as js_file:
    #     json.dump(dataset, js_file, indent=13)

    # with open('./inputs/dataset_test_noise.js', 'w') as js_file:
    #     json.dump(dataset, js_file, indent=13)

# =========================================================================================
    # plot distribution histogram for every feature
    #
    # plot_diagram(gender, "gender")
    # plot_diagram(occupation, "occupation")
    # plot_diagram(educational_qualification, "education qualification")
    # plot_diagram(income, "income")
    # plot_diagram(installment, "installment")
    # plot_diagram(arrears, "arrears")
    # plot_diagram(deposit, "deposit")
    # plot_diagram(shopping_frequency, "shopping frequency")
    # plot_diagram(average_cost, "average cost")
    # plot_diagram(ages, "ages")
    # plot_diagram(number_of_browsing, "number browsing per day")
    # plot_diagram(number_of_raise, "people to raise")
    # plot_diagram(passive_income, "passive income")

# =========================================================================================
    # save dataset to csv file
    # df = pd.DataFrame(dataset)
    # df.to_csv('./dataset_test_noise.csv', index=False)
