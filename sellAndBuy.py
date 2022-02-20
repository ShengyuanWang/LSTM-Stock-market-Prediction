# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ optimal_stock_action.py ]
#   Synopsis     [ Best Time to Buy and Sell Stock with Transaction Fee - with Dynamic Programming ]
#   Author       [ Ting-Wei Liu (Andi611) ]
#   Copyright    [ Copyleft(c), NTUEE, NTU, Taiwan ]
"""*********************************************************************************************"""

###############
# IMPORTATION #
###############
import sys
import operator
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def find_optimal_action(priceVec, transFeeRate, use_DP=True):
    _BUY = 1
    _HOLD = 0
    _SELL = -1

    dataLen = len(priceVec)
    actionVec = np.zeros(dataLen)

    # Dynamic Programming method
    if use_DP:
        capital = 1
        money = [{'money': 0, 'from': 0} for _ in range(dataLen)]
        stock = [{'stock': 0, 'from': 1} for _ in range(dataLen)]

        # DP initialization
        money[0]['money'] = capital
        stock[0]['stock'] = capital * (1 - transFeeRate) / priceVec[0]

        # DP recursion
        for t in range(1, dataLen):

            # find optimal for sell at time t:
            hold = money[t - 1]['money']
            sell = stock[t - 1]['stock'] * priceVec[t] * (1 - transFeeRate)

            if hold > sell:
                money[t]['money'] = hold
                money[t]['from'] = 0
            else:
                money[t]['money'] = sell
                money[t]['from'] = 1

            # find optimal for buy at time t:
            hold = stock[t - 1]['stock']
            buy = money[t - 1]['money'] * (1 - transFeeRate) / priceVec[t]

            if hold > buy:
                stock[t]['stock'] = hold
                stock[t]['from'] = 1
            else:
                stock[t]['stock'] = buy
                stock[t]['from'] = 0

        # must sell at T
        prev = 0
        actionVec[-1] = _SELL

        # DP trace back
        record = [money, stock]
        for t in reversed(range(1, dataLen)):
            prev = record[prev][t]['from']
            actionVec[t - 1] = _SELL if prev == 0 else _BUY

        # Action smoothing
        prevAction = actionVec[0]
        for t in range(1, dataLen):
            if actionVec[t] == prevAction:
                actionVec[t] = _HOLD
            elif actionVec[t] == -prevAction:
                prevAction = actionVec[t]

        return actionVec

    # Baseline method
    else:
        conCount = 3
        for ic in range(dataLen):
            if ic + conCount + 1 > dataLen:
                continue
            if all(x > 0 for x in
                   list(map(operator.sub, priceVec[ic + 1:ic + 1 + conCount], priceVec[ic:ic + conCount]))):
                actionVec[ic] = _BUY
            if all(x < 0 for x in
                   list(map(operator.sub, priceVec[ic + 1:ic + 1 + conCount], priceVec[ic:ic + conCount]))):
                actionVec[ic] = _SELL
        prevAction = _SELL

        for ic in range(dataLen):
            if actionVec[ic] == prevAction:
                actionVec[ic] = _HOLD
            elif actionVec[ic] == -prevAction:
                prevAction = actionVec[ic]
        return actionVec


def profit_estimate(priceVec, transFeeRate, actionVec):
    capital = 1
    capitalOrig = capital
    dataCount = len(priceVec)-10
    suggestedAction = actionVec

    stockHolding = np.zeros((dataCount))
    total = np.zeros((dataCount))

    total[0] = capital
    print(dataCount)
    for ic in range(dataCount):
        currPrice = priceVec[ic]
        if ic > 0:
            stockHolding[ic] = stockHolding[ic - 1]
        if suggestedAction[ic] == 1:
            if stockHolding[ic] == 0:
                stockHolding[ic] = capital * (1 - transFeeRate) / currPrice
                capital = 0

        elif suggestedAction[ic] == -1:
            if stockHolding[ic] > 0:
                capital = stockHolding[ic] * currPrice * (1 - transFeeRate)
                stockHolding[ic] = 0

        elif suggestedAction[ic] == 0:
            pass
        else:
            assert False
        total[ic] = capital + stockHolding[ic] * currPrice * (1 - transFeeRate)


    returnRate = (total[-1] - capitalOrig) / capitalOrig
    return returnRate


if __name__ == '__main__':
    SEARCH = False

    # try:
    #     df = pd.read_csv(sys.argv[1])
    #     transFeeRate = float(sys.argv[2])
    # except:
    #     df = pd.read_csv('./data/BCHAIN-MKPRU.csv')
    #     transFeeRate = float(0.01)
    df1 = pd.read_csv('./code/data/BCHAIN-MKPRU.csv')
    transFeeRate1 = float(0.02)
    df2 = pd.read_csv('./code/data/gold_data.csv')
    transFeeRate2 = float(0.01)
    df11 = pd.read_csv('./code/prediction/bitcoin_LSTM_result_modified.csv')
    df22 = pd.read_csv('./code/prediction/gold_LSTM_result_modified.csv')
    # df11 = df11.drop([df11.columns[0]], axis=1)
    # df22 = df22.drop([df22.columns[0]], axis=1)
    priceVec1 = df1["Value"].values
    priceVec2 = df2["USD (PM)"].values
    pricePre1 = df11["value"].values
    pricePre2 = df22["value"].values
    print('Optimizing over %i numbers of transactions.' % (len(priceVec1)))

    actionVec1 = find_optimal_action(pricePre1, transFeeRate1)
    returnRate1 = profit_estimate(priceVec1, transFeeRate1, actionVec1)
    actionVec2 = find_optimal_action(pricePre2, transFeeRate2)
    returnRate2 = profit_estimate(priceVec2, transFeeRate2, actionVec2)
    rate = [0.812, 0.187]
    returnRate = rate[0]*returnRate1+rate[1]*returnRate2
    actionVec2[0]= 0
    actionVec1[0] = 0
    print('Return rate: ', returnRate)
    plt.figure(figsize=[20, 10])
    plt.plot(df1['Value'], label= 'BitCoin_price')
    plt.plot(df11['value'], label='BitCoin_price_Predict')
    buy = False
    sell = False
    for i in range(len(actionVec1)):
        if actionVec1[i] == 1:
            if buy == False:
                plt.plot(i, df1['Value'][i],
                     '^', ms=10, label='Buy Signal', color='red')
            else:
                plt.plot(i, df1['Value'][i],
                         '^', ms=10, color='red')
            buy = True
    for i in range(len(actionVec1)):
        if actionVec1[i] == -1:
            if sell == False:
                plt.plot(i, df1['Value'][i],
                     'o', ms=10, label='Sell Signal', color='blue')
            else:
                plt.plot(i, df1['Value'][i],
                         'o', ms=10,  color='blue')
            sell = True

    plt.legend()
    plt.grid()
    plt.show()
    plt.figure(figsize=[20, 10])
    plt.plot(df2["USD (PM)"], label='gold_price')
    plt.plot(df22['value'], label='gold_price_Predict')
    buy = False
    sell = False
    for i in range(len(actionVec2)):
        if actionVec2[i] == 1:
            if buy == False:
                plt.plot(i, df2["USD (PM)"][i],
                         '^', ms=15, label='Buy Signal', color='red')
            else:
                plt.plot(i, df2["USD (PM)"][i],
                         '^', ms=15, color='red')
            buy = True
    for i in range(len(actionVec2)):
        try:
            if actionVec2[i] == -1:
                if sell == False:
                    plt.plot(i, df2['USD (PM)'][i],
                             'o', ms=15, label='Sell Signal', color='blue')
                else:
                    plt.plot(i, df2['USD (PM)'][i],
                             'o', ms=15, color='blue')
                sell = True
        except:
            print(i)
    plt.legend()
    plt.grid()
    plt.show()



