import pandas as pd

URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/taxis.csv"

def load_data():
    df = pd.read_csv(URL)
    df.to_csv("data/raw/data.csv", index=False)
    print("✅ Data saved to data/raw/data.csv")

if __name__ == "__main__":
    load_data()


