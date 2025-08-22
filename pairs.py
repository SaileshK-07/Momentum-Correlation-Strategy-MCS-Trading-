import os
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

N = 50
T = 60
P = 7
R = 0.8
M = 4
F = 0.005
D = 500
B = 10000

def GetData(file_path: str) -> pd.DataFrame:
    """Load the price dataset with only required columns."""
    return pd.read_csv(file_path, usecols=["datadate", "tic", "adjcp"])

def PartitionData(data: pd.DataFrame):
    """Split the dataframe by date and create an index mapping."""
    date_to_index = {}
    for idx, date in enumerate(data["datadate"]):
        date_to_index.setdefault(str(date), idx)
    return [np.array_split(data, 2926), date_to_index]

def GetMomentumBasedPriority(partitioned_data, date_to_index, today):
    """Calculate momentum weights for the given day."""
    n_days_ago = (
        datetime.date(int(today[0:4]), int(today[4:6]), int(today[6:]))
        + datetime.timedelta(days=-N)
    )

    i = 0
    while True:
        candidate = str(n_days_ago - datetime.timedelta(days=i)).replace("-", "")
        if candidate in date_to_index:
            break
        i += 1

    temp = candidate

    today_idx = date_to_index[today]
    past_idx = date_to_index[temp]

    momentum = (
        np.array(partitioned_data[today_idx]["adjcp"])
        - np.array(partitioned_data[past_idx]["adjcp"])
    )

    summation = np.array(partitioned_data[today_idx]["adjcp"], dtype=float)
    for j in range(past_idx + 1, today_idx):
        summation += np.array(partitioned_data[j]["adjcp"], dtype=float)

    return momentum * N / summation

def GetBalanced(prices, weights, balance):
  copy = np.flip(np.sort(weights))
  for i in range(M,len(weights)):
    copy[i] = 0
  for i in range(len(weights)):
    if weights[i] not in copy:
      weights[i] = 0
    elif weights[i] < 0:
      weights[i] = 0
  sum = np.sum(weights)
  if (sum <= 0):
    return np.zeros(30, dtype = float)
  weights /= sum
  sum = np.sum(weights * prices)
  return (balance / sum) * weights

def Switch(firstStock, secondStock, today ,PartitionedDataFrames):
  weights = GetMomentumBasedPriority(PartitionedDataFrames, DateToIndex, today)
  if weights[firstStock] >= weights[secondStock]:
    return firstStock
  else:
    return secondStock
  
class PortFolio:
    def __init__(self, balance, num_stocks, neg_corr, prices):
        self.balance = balance
        self.numStocks = num_stocks
        self.numStocks += GetBalanced(
            prices,
            GetMomentumBasedPriority(
                PartitionedData, DateToIndex, list(DateToIndex.keys())[T]
            ),
            self.balance,
        )
        self.prices = prices
        self.balance -= np.sum(self.numStocks * self.prices) * (1 + F)
        self.negCorr = neg_corr

    def SellStock(self, index):
        self.balance += self.numStocks[index] * self.prices[index] * (1 - F)
        self.numStocks[index] = 0

    def BuyStock(self, index, number):
        self.balance -= number * self.prices[index] * (1 + F)

    def CalculateNetWorth(self):
        return self.balance + np.sum(self.numStocks * self.prices) * (1 - F)

    def ChangePricesTo(self, newPriceVector):
        self.prices = newPriceVector

    def ChangePairs(self):
        if self.negCorr == []:
            return
        for i in range(len(self.negCorr) - 1):
            x = Switch(self.negCorr[i][0], self.negCorr[i][1], today, PartitionedData)
            y = self.negCorr[i][0] + self.negCorr[i][1] - x
            credit = self.numStocks[y] * self.prices[y] * (1 - F)
            self.numStocks[y] = 0
            self.numStocks[x] += credit / (self.prices[x] * (1 + F))
  
    def RebalancePortfolio(self):
        TdaysAgo = datetime.date(int(today[0:4]), int(today[4:6]), int(today[6:])) + datetime.timedelta(days=-T)
        i = 0
        while True:
            x = str(TdaysAgo - datetime.timedelta(days=i)).replace("-", "")
            if x in DateToIndex.keys():
                break
            i += 1
        temp = x
        Sum = [np.array(PartitionedData[DateToIndex[today]]["adjcp"])]
        for j in range(DateToIndex[temp] + 1, DateToIndex[today]):
            Sum.append(np.array(PartitionedData[j]["adjcp"]))
        corrCoef = np.corrcoef(np.array(Sum))
        copy = np.sort(corrCoef.flatten())
        for k in range(M):
            for i in range(corrCoef.shape[0]):
                for j in range(30):
                    if abs(corrCoef[i][j] - copy[k]) < 1e-10:
                        self.negCorr.append((i, j))
        self.negCorr = self.negCorr[0:M]
        copy = self.negCorr.copy()
        if self.negCorr == []:
            return
        for i in range(len(self.negCorr) - 1):
            for j in range(i + 1, len(self.negCorr)):
                x = list(self.negCorr[i])
                y = list(self.negCorr[j])
                if x[0] == y[0] or x[0] == y[1] or x[1] == y[0] or x[1] == y[1]:
                    if self.negCorr[j] in copy and copy.count(self.negCorr[j]) > 1:
                        copy.remove(self.negCorr[j])
        self.negCorr = copy.copy()
        self.ChangePairs()
    
def VisualizeData(FinalData):
  plt.plot(FinalData)
  plt.show()

DATA_FILE = os.path.join(os.path.dirname(__file__), "DATA.csv")
Data = GetData(DATA_FILE)
List = PartitionData(Data)
PartitionedData = List[0]
DateToIndex = List[1]

myPortfolio = PortFolio(
    B * R,
    np.zeros(30, dtype=float),
    [],
    np.array(PartitionedData[T]["adjcp"]),
)
NetWorthAfterEachTrade = [myPortfolio.CalculateNetWorth() + B * (1 - R)]

for i in range(T + 1, len(PartitionedData)):
    today = list(DateToIndex.keys())[i]
    myPortfolio.ChangePricesTo(np.array(PartitionedData[i]["adjcp"]))
    NetWorthAfterEachTrade.append(myPortfolio.CalculateNetWorth() + B * (1 - R))
    if i % T == 0:
        myPortfolio.RebalancePortfolio()
    if i % P == 0:
        myPortfolio.ChangePairs()
    if i == T + D + 6:
        break

VisualizeData(NetWorthAfterEachTrade[:D])

