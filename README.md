# Data_Projects

A collection of data analysis and machine learning projects in Python and R.

| Project | Language | Type | Summary |
|---|---|---|---|
| [The Office](Office/) | Python | Analysis | Did Michael Scott's exit change the show's viewership, ratings and character dynamics? |
| [Elections](Elections/) | R | ML / Kaggle | Predicting county-level Democratic vote share. **1st of 20 teams** |
| [Heart](Heart/) | R | ML / Kaggle | Predicting heart disease from patient data. **2nd of 20 teams** |
| [Google Play Store](GoogleAppStore/) | Python | EDA | What drives app installs and ratings, across 10,000+ apps |

---

## The Office: Michael Scott's Impact
Combines a Kaggle dataset of every line of dialogue with episode data scraped from Wikipedia and IMDb ratings.

- **Viewership:** an interrupted time-series regression shows a drop of about 1.1M viewers at Michael's exit, followed by a faster decline.
- **Character network:** scene co-appearance networks show the cast's interactions became less concentrated around a few characters after he left. Bootstrap intervals support this.

Files: `01_data_prep.ipynb` (scraping, cleaning, merging) → `02_analysis.ipynb` (regression, network analysis)
Tools: pandas, BeautifulSoup, statsmodels, NetworkX, Plotly

## Elections: Predicting Vote Share
Kaggle competition for Stats 101C (Summer 2022). We predicted each county's Democratic vote share from census demographic features, using a stacked ensemble of XGBoost, SVM (radial), glmnet and random forest models built with `caret` / `caretEnsemble`.
**Result: 1st out of 20 teams.**

## Heart: Predicting Heart Disease
Kaggle competition for Stats 101C (Summer 2022). We predicted heart disease from clinical measurements (age, cholesterol, chest pain type, etc.). The approach used KNN imputation for missing values and a `tidymodels` stacked ensemble of random forest, neural net, XGBoost and KNN models, tuned with repeated 10-fold cross-validation.
**Result: 2nd out of 20 teams.**

## Google Play Store: Exploratory Data Analysis
Cleaning and exploring 10,000+ apps to see which factors relate to installs and ratings: category, price, content rating and genre. Data: [Kaggle](https://www.kaggle.com/lava18/google-play-store-apps)
Tools: pandas, matplotlib, seaborn

---

## Running the projects
- **Python:** `pip install pandas numpy matplotlib seaborn statsmodels scipy networkx plotly kaleido requests beautifulsoup4`, then run the notebooks in order.
- **R:** install `tidyverse`, `tidymodels`, `caret`, `caretEnsemble`, `stacks`, `xgboost`, `ranger` and `kknn`, then run the script in each folder.
