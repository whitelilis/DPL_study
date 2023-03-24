#!/usr/bin/env python -i

import pandas
import sys
import matplotlib.pyplot as plt
plt.ion() # IMPORT: for plt.show() not hung the terminal


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



def tick_df(argv):
    if len(argv) < 2:
        #print(f"Usage: python3 {argv[0]} <parquet file> <symbol1> [symbol2] ...")
        return None
    else:
        df = pandas.read_parquet(argv[0])
        ret = []
        for x, s in enumerate(argv[1:]):
            df_tmp = df[df.symbol == s]
            #ret.append(clear(df_tmp))
            ret.append(df_tmp)
            step, price_range = symbol_price_range(df_tmp)
            print("%2d %7s volume: %7d, step %9.3f range %10d" % (x, s, df_tmp.volume.max() - df_tmp.volume.min(), step, price_range))
    return ret



draw_columns = ['bidVolume5', 'bidVolume4', 'bidVolume3', 'bidVolume2', 'bidVolume1', 'askVolume1', 'askVolume2', 'askVolume3', 'askVolume4', 'askVolume5']
columns = draw_columns
columns.append('lastPrice')
def flow_one_tick(df, indexes):

    aim_df = df.iloc[indexes]
    aim_df = aim_df[columns]

    aim_df = aim_df.reset_index()
    aim_df.drop(columns=['index'], inplace=True)
    return aim_df

def price2index(x_min, price, step):
    return int((price - x_min) / step)

def draw_one_row(row, x_min, x_max, step):
    last_green_index = price2index(x_min, row.bidPrice1, step)
    left_size = last_green_index + 1
    right_size = price2index(x_min, x_max, step) - last_green_index

    print(f"----------> max is {x_max}, min is {x_min}, left is {left_size}, right is {right_size} <------------")

    green_colors = ['green'] * left_size
    red_colors = ['red'] * right_size
    all_colors = green_colors + red_colors
    left_data = [0] * left_size
    right_data = [0] * right_size

    left_data[-1] = row.bidVolume1
    left_data[-2] = row.bidVolume2
    left_data[-3] = row.bidVolume3
    left_data[-4] = row.bidVolume4
    left_data[-5] = row.bidVolume5

    right_data[0] = row.askVolume1
    right_data[1] = row.askVolume2
    right_data[2] = row.askVolume3
    right_data[3] = row.askVolume4
    right_data[4] = row.askVolume5

    data_all = left_data + right_data

    labels = []
    d_start = x_min
    while d_start <= x_max:
        labels.append(f"{d_start}")
        d_start += step

    plt.clf()
    plt.bar(labels, data_all, color=all_colors)
    plt.pause(0.05)



def draw_flow(df, step):
    x_min = min(df.lastPrice)
    x_max = max(df.lastPrice)
    df.apply(lambda row: draw_one_row(row, x_min - 5*step, x_max + 5*step, step), axis=1) # apply is fast than df.itertupples()

def sim(df):
    ret = df[["lastPrice", "volume", "askPrice1", "askVolume1", "bidPrice1", "bidVolume1"]]
    ret.columns = ['p', 'v', 'a1', 'av1', 'b1', 'bv1']
    ret.loc[:,'isb'] = ret.apply(lambda row: row['p'] == row['b1'], axis=1)
    ret.loc[:,'dv'] = ret.v.diff()
    return ret[1:]

if __name__ == "__main__":
    dfs = tick_df(sys.argv[1:])
    t1 = dfs[0]
    