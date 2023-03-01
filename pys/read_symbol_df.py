#!/usr/bin/env python -i

import pandas
import sys
import matplotlib.pyplot as plt




def time_invalidate(time_str):
    return ( '02:30' <= time_str <= '09:00' ) or ( '11:30' <= time_str <= '13:00' ) or ('15:15' <= time_str <= '21:00' )


def clear(df):
    df.loc[:'nn1'] = df.updateTime.apply(time_invalidate)
    ret = df[df.nn1 == False]
    return ret




def symbol_price_range(df):
    return 0, 0
    first_count = 10
    ask1_sum = sum(list(df.askPrice1)[:first_count])
    bid1_sum = sum(list(df.bidPrice1)[:first_count])
    step = (bid1_sum - ask1_sum) / first_count
    first_price = list(df.lastPrice)[0]
    last_price = list(df.lastPrice)[-1]

    price_range = (last_price - first_price) / step
    return step, price_range



def tick_df():
    if len(sys.argv) < 3:
        print(f"Usage: python3 {sys.argv[0]} <parquet file> <symbol1> [symbol2] ...")
        exit(1)
    else:
        df = pandas.read_parquet(sys.argv[1])
        ret = []
        for x, s in enumerate(sys.argv[2:]):
            df_tmp = df[df.symbol == s]
            #ret.append(clear(df_tmp))
            ret.append(df_tmp)
            step, price_range = symbol_price_range(df_tmp)
            print("%2d %7s volume: %7d, step %9.3f range %10d" % (x, s, df_tmp.volume.max() - df_tmp.volume.min(), step, price_range))
    return ret

if __name__ == "__main__":
    dfs = tick_df()
