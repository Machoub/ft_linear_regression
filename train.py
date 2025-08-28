import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from predict import estimate_price

def standardize_data(mileages, mean, std):
    standardized = (mileages - mean) / std
    return standardized

def train_model(learn_rate, n_iterations, data):
    theta0 = 0
    theta1 = 0
    m = len(data)

    for i in range(n_iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for mileage, price in data:
            error = estimate_price(mileage, theta0, theta1) - price
            tmp_theta0 += error
            tmp_theta1 += error * mileage 
        theta0 -= learn_rate * (1 / m) * tmp_theta0
        theta1 -= learn_rate * (1 / m) * tmp_theta1
    return theta0, theta1


def main():
    dataset = np.genfromtxt("data.csv", delimiter=',', skip_header=1, filling_values=np.nan)
    km = dataset[:, 0] # x
    price = dataset[:, 1] # y
    
    x_mean = np.mean(km)
    x_std = np.std(km)

    km_standardized = standardize_data(km, x_mean, x_std)
    list_standardized = list(zip(km_standardized, price))

    learn_rate = 0.1
    n_iterations = 1000
    theta0, theta1 = train_model(learn_rate, n_iterations, list_standardized)
    # Print the trained parameters
    print("Learning done! Here are the trained parameters:")
    print("Theta0 =", theta0)
    print("Theta1 =", theta1)

    with open("thetas.txt", "w") as f:
        f.write(f"Theta0 = {theta0}\n")
        f.write(f"Theta1 = {theta1}\n")

if __name__ == "__main__":
    main()
