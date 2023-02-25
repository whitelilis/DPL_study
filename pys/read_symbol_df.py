#!/usr/bin/env python -i

import pandas
import sys
import matplotlib.pyplot as plt


def tick_df():
    if len(sys.argv)!= 3:
        print("Usage: python3 {} <parquet file> <symbol>".format(sys.argv[0]))
        exit(1)
    else:
        df = pandas.read_parquet(sys.argv[1])
        return df[df.symbol == sys.argv[2]]

if __name__ == "__main__":
    df = tick_df()
