### Algorithmic Trading Self-Project

#### Screenshots
- Paired Switching Result
- The idea behind this strategy is that if the assets are negatively correlated, then a traditional mixed portfolio might lead to a lower return than the return for the individual assets.
- We periodically compute the Co-relation Coefficients' Matrix and choose pair of stocks which have the most negative co-relations and trade in them. The basic idea being that if one of them would fall, then the other would rise and hence, we must switch between them!
  <img width="480" alt="Screenshot 2024-06-19 at 11 48 23 AM" src="https://github.com/Deep99739/algotrading/assets/142610132/32290a50-4770-4d09-ba26-98474bb5dc09">
  
- Momentum Result
  
- Momentum-based Trading is based on the assumption that Stocks which have performed will in the past, will perform better in the coming future.
- Momentum(For A particular stock) = Close Price(Today) - Close Price(N-day ago)
- Hyper-parameters:
- N: Number of days to look back to calculate momentum.
- T: Time interval after which the portfolio is reshuffled.
- R: Ratio of total balance reserved for risk-free assets.
- M: Number of top stocks (based on momentum) to include in the portfolio.
- F: Brokerage fee, typically less than 0.5% of the total transaction amount.
  These hyper-parameters are crucial in defining and optimizing a momentum-based trading strategy.
  
  <img width="463" alt="Screenshot 2024-06-19 at 11 48 57 AM" src="https://github.com/Deep99739/algotrading/assets/142610132/b2f67aa1-8c98-420f-b1ce-05ff7dbecf1a">
#### Requirements
- Preferred OS to run: Linux
- Packages:
  - python3 (install: Linux, OSX)
  - pygame (install: Linux/OSX)
- In case of any installation issues, visit [here](link to troubleshooting guide).

#### Instructions

**Setup:**
1. Ensure you have Python 3 installed.
2. Install pygame using the appropriate package manager for your OS.

**Execution:**
- From the project directory, execute one of the following commands in your Terminal:
    ```bash
    python3 momentum.py
    python3 pairs.py
    ```

**Data Path Update:**
- **Important:** Before running `momentum.py`, `momentum2.ipynb`, `pairs.py`, or `pairs2.ipynb`, ensure you update the path to the `data.csv` file in each of these files. Locate the line where `data.csv` is loaded and update it with the correct path on your system.

**Alternative Approach:**
- Alternatively, you can open this folder in Jupyter Notebook and run:
    - `momentum2.ipynb` instead of `momentum.py`
    - `pairs2.ipynb` instead of `pairs.py`
- This approach can simplify data path management as Jupyter Notebook allows for relative paths.
