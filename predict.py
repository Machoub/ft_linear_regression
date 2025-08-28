import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage) #y^

def main():
    try:
        
        mileage = float(input("Enter the mileage of the car: "))
        if not isinstance(mileage, (int, float)):
            print("Invalid input. Please enter a numeric value.")
            return
        if mileage < 0:
            print("Mileage cannot be negative. Please enter a valid mileage.")
            return
        
        theta0 = 0
        theta1 = 0
        if os.path.exists("thetas.txt"):
            with open("thetas.txt", "r") as f:
                lines = f.readlines()
                theta0 = float(lines[0].strip().split('=')[1])
                theta1 = float(lines[1].strip().split('=')[1])

        dataset = np.genfromtxt("data.csv", delimiter=',', skip_header=1, filling_values=np.nan)
        km = dataset[:, 0] #x
        price = dataset[:, 1] #y

        mean_km = np.mean(km)
        std_km = np.std(km)

        km_standardized = (mileage - mean_km) / std_km
        estimated_prices = estimate_price(km_standardized, theta0, theta1)

        # Print result
        print("estimated price for a car with a mileage of {:.0f} km: {:.2f} $"
              .format(mileage, estimated_prices))
        
        if theta0 == 0 and theta1 == 0:
            print("Model parameters not found. Please train the model first using train.py.")
            return
        #calculates the precision
        rmse = np.sqrt(np.mean((estimate_price(km_standardized, theta0, theta1) - price) ** 2))
        mae = np.mean(np.abs(estimate_price(km_standardized, theta0, theta1) - price))
        R_2 = 1 - (np.sum((price - estimate_price(km_standardized, theta0, theta1)) ** 2) / np.sum((price - np.mean(price)) ** 2))
        print(f"RMSE : {rmse:.2f}")
        print(f"MAE  : {mae:.2f}")
        print(f"R2   : {R_2:.2f}")

        # Plotting
        plt.scatter(km, price, color='blue', label='Data points', zorder=2)
        plt.plot(km, estimate_price((km - mean_km) / std_km, theta0, theta1), color='red', label='Linear Regression', zorder=1)
        plt.scatter(mileage, estimated_prices, color='green', label='Prediction', zorder=3)
        plt.xlabel('Mileage')
        plt.ylabel('Price')
        plt.title(
            "Estimated price of {:.0f} km: {:.2f} $"
            .format(mileage, estimated_prices))
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        print("Process interrupted.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()