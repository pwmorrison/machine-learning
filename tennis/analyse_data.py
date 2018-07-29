import os
import numpy as np
import pandas as pd

def main():
    filename = r"C:/Users/Paul/tennis_atp/atp_matches_2018.csv"
    df = pd.read_csv(filename)
    print(df.describe())
    print(df.head)


if __name__ == '__main__':
    main()
