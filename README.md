# Lingjun_China_A-Share_Market_Microstructure_Prediction

## Overview

Welcome to the 2026 Lingjun China A-Share Market Microstructure Prediction Challenge!

In the fast-paced world of quantitative trading, the ability to extract signal from noise at the intraday level is what separates successful strategies from the rest. This year, we shift our focus from futures to the China A-share market, a unique financial landscape characterized by high liquidity, significant retail participation, and rich microstructure dynamics.

Participants are tasked with building predictive models using high-fidelity, aggregated Level 2 market data. Unlike traditional daily price forecasting, this competition challenges you to model price movements over short-to-medium intraday horizons using anonymous features derived from tick-by-tick order books and transaction flows.

The core challenge lies in navigating the "curse of dimensionality"—with over hundreds of features per timestamp—while accounting for the non-stationary nature of financial time series and the heavy-tailed distributions typical of high-frequency data. Can you decode the hidden patterns in the order book and order/trade flows to forecast the next move in one of the world's most vibrant equity markets?

## Description

### The Pulse of the Order Book

Modern quantitative finance has moved beyond simple OHLC charts. Today, the "edge" resides in market microstructure: the study of how individual trades, limit orders, and cancellations interact to form price discovery. In this competition, you will work with a massive dataset covering hundreds of China A-share stocks, featuring information distilled from the exchange’s raw tick-by-tick feed.

### The Dataset

We have provided several hundred days of historical data for a diverse universe of 500 stocks. The features are engineered into minute bars, capturing market microstructure dynamics from various perspectives within the minute, allowing sequential models to presumably have an edge in this competition. The targets are longer than a minute and strictly intraday - we do not ask you to predict overnight gaps.

train.parquet:
stockid: 0-499, dateid: 0-359, timeid: 0-238, f0-f383, LabelA is target
{'stockid': [0, 0, 0, 0, 0],
 'dateid': [0, 0, 0, 0, 0],
 'timeid': [0, 1, 2, 3, 4],
 'exchangeid': [0, 0, 0, 0, 0],
 'f0': [0.0,
  0.1701408177614212,
  0.0977163016796112,
  0.09508412331342697,
  0.08755901455879211],
 'f1': [0.8276021480560303,
  1.5028387308120728,
  2.475088119506836,
  1.8099485635757446,
  1.8630694150924683],
 'f2': [nan,
  1.0367116928100586,
  1.3906992673873901,
  0.501453697681427,
  1.0773588418960571],
 'f3': [nan,
  -0.2709181606769562,
  0.18470048904418945,
  0.023560311645269394,
  0.2590816020965576],
 'f4': [8.091544347221613e-39,
  0.1618461310863495,
  0.13262063264846802,
  0.1553269773721695,
  0.29446569085121155],
...
 'LabelA': [-0.02396014705300331,
  -0.01811475306749344,
  -0.008410797454416752,
  -0.011145544238388538,
  -0.012443514540791512],
 'LabelB': [-0.03326994925737381,
  -0.024314910173416138,
  -0.010967634618282318,
  -0.016765035688877106,
  -0.015333488583564758],
 'LabelC': [-0.028307124972343445,
  -0.021935824304819107,
  -0.018478985875844955,
  -0.02259679138660431,
  -0.01990014873445034]}

test.parquet:

{'stockid': [0, 0, 0, 0, 0],
 'dateid': [0, 0, 0, 0, 0],
 'timeid': [0, 1, 2, 3, 4],
 'exchangeid': [0, 0, 0, 0, 0],
 'f0': [nan,
  0.14547176659107208,
  0.12428752332925797,
  0.04504920542240143,
  0.06788093596696854],
 'f1': [nan, 1.0684348344802856, 0.966545045375824, 0.0, 0.0],
 'f2': [nan, 1.118942379951477, 0.928851306438446, nan, nan],
 'f3': [nan, -1.185532569885254, 1.1293635368347168, nan, nan],
 'f4': [0.0,
  0.5408017039299011,
  0.6422995924949646,
  3.097761691606138e-08,
  2.8250492725306355e-35],
 'f5': [0.005750067066401243,
  0.4395080506801605,
  0.4095383584499359,
  0.0,
  0.40476223826408386],
...
 'f383': [nan,
  0.6404368877410889,
  0.1271507292985916,
  -0.12024550139904022,
  0.16935458779335022]}

### The Challenge: Intraday Multi-Horizon Forecasting

The primary objective is to predict a specific target return. To assist in the regularization of your models, we have provided auxiliary labels representing different time horizons. Your goal is to develop a robust algorithm that can:

1. Generalize across hundreds of different stocks with varying liquidity profiles.

2. Adapt to sudden shifts in market regime and volatility.

3. Extract predictive value from a high-dimensional feature space without succumbing to noise.

### Why Participate?

By participating, you will tackle the same obstacles faced by top-tier quantitative hedge funds and market makers. Whether you are a seasoned "quant" or a data scientist looking to break into finance, this competition offers a rare opportunity to test your skills on high-quality, real-world exchange data that is often inaccessible to the public.

## Evaluation

### Metric

Submissions are evaluated on the R-Squared of the predicted returns：

Successful models will demonstrate an ability to capture significant market dislocations. In high-frequency trading, predicting the direction of a low-volatility asset is often less critical than correctly sizing the move of a high-volatility asset. The evaluation metric reflects this by rewarding models that maintain accuracy during periods of high market activity.

### Submission File

For each ID in the test set, you must predict market returns. The file should contain a header and have the following format:

stockid|dateid|timeid,prediction

0|0|0,0

0|0|1,0

The first column represents a unique index separated by pipe without spaces and contains information of stockid, dateid, timeid. The second column is your return prediction.

### Important Note

1. We do not evaluate your predictions of the last 10 timeids (from 229 to 238) though predictions of these timeids still need to be uploaded to make your submission a valid file for evaluation.

2. For any timeid required to make a prediction, you can ONLY use data at that dateid and timeid or any dateid and timeid before (i.e. no forward looking, though available data is not limited to any single stock, should you find data from other stocks is helpful).

3. Your team must provide the source code to us in order to be considered as a successful final submission - please make sure that we are able to reproduce your results and be ready to explain your thoughts during the dissertation phase ;)
