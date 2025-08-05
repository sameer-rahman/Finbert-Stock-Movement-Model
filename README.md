# Stock Movement Prediction using FinBERT and Structured Features

## Overview

This project predicts next-day stock price movements using a combination of natural language processing (NLP) and structured financial data.

We use FinBERT embeddings derived from daily aggregated stock headlines, along with engineered features such as dollar volume, sentiment score, and weekday, to predict whether a stock's price will increase by more than 2.5% the following day.

### File Structure

- **`final_model_pipeline.ipynb`** – Main modeling notebook containing:
  - Neural network architecture using FinBERT + structured features
  - Probability calibration
  - Precision-recall and ROC analysis
  - Comparisons against LightGBM and logistic regression baselines

- **`download_data.ipynb`** – Data loading and preprocessing steps

### Final Model Performance

- **F1 Score:** ~0.283  
- **AUC:** ~0.608  
- The final model demonstrates high recall and moderate precision, outperforming random and baseline models.

## Dataset

* Aggregated daily news headlines per stock
* Historical price data retrieved using the Yahoo Finance API
* Final dataset contains one row per (stock, date) with FinBERT embeddings and structured features

## Target Variable

Binary target indicating whether a stock’s next-day return exceeds 2.5%:

```python
(target = price_return > 0.025).astype(int)
```

## Model Comparison Summary

### MLP-Based Models

| Model Variant                   | AUC    | F1     | Precision | Recall |
| ------------------------------- | ------ | ------ | --------- | ------ |
| FinBERT Embeddings Only         | 0.6059 | 0.2683 | 0.172     | 0.619  |
| Structured Features Only        | 0.6052 | 0.2687 | 0.172     | 0.615  |
| Top 400 FinBERT + Features (Best) | 0.6094 | 0.2837 | 0.188     | 0.591  |
| Combined (No Ticker Embedding)  | 0.5740 | 0.2412 | 0.158     | 0.448  |
| Combined + Ticker Embedding     | 0.6064 | 0.2837 | 0.182     | 0.498  |

### Baselines

| Baseline Model      | AUC   | F1     | Precision | Recall | Best Threshold |
| ------------------- | ----- | ------ | --------- | ------ | -------------- |
| Dummy Classifier    | 0.500 | 0.2349 | 0.133     | 1.000  | 0.500          |
| Logistic Regression | 0.553 | 0.2434 | 0.156     | 0.550  | 0.139          |

## Final Model: Multi-Layer Perceptron

* Input: top 8 FinBERT embedding dimensions (selected using SHAP) + 7 engineered features
* Includes 8-dimensional ticker embedding
* Architecture: \[Input → Linear(256) → ReLU → Dropout → Linear(128) → ReLU → Dropout → Output]
* Trained with binary cross-entropy loss
* Upsampled training data for balanced learning
* Threshold for classification selected via F1 score sweep

## Feature Selection via SHAP

We trained a LightGBM model on all 768 FinBERT dimensions and used SHAP to identify the top 8 most informative. These 8 were used in the final model to reduce dimensionality and mitigate overfitting.

## Example Predictions

Top predictions with highest confidence scores (> 99%):

| Date       | Ticker | Confidence | True Label |
| ---------- | ------ | ---------- | ---------- |
| 2020-06-23 | REG    | 0.9997     | 1          |
| 2022-12-13 | NWS    | 0.9992     | 1          |
| 2022-03-25 | NOV    | 0.9991     | 1          |

## Threshold Optimization

To handle class imbalance, we sweep over 200 thresholds and evaluate F1, precision, and recall to select the optimal threshold.

## Structured Features Used

* `headline_count`: Number of headlines that day
* `sentiment_volume`: Sum of sentiment × count
* `day_of_week`: Integer day (0 = Monday)
* `avg_sentiment`: Average FinBERT sentiment
* `is_q_earnings_month`: 1 if Jan, Apr, Jul, Oct
* `dollar_volume`: Close × volume
* `prev_day_return`: Daily return from t-1

## Key Insights

* FinBERT embeddings alone provide high recall but low precision
* Structured features alone match FinBERT on AUC and F1
* Combining FinBERT + structured data yields best performance
* Ticker embeddings improve generalization
* SHAP pruning helped isolate signal-carrying FinBERT dimensions

## Future Improvements

* Try LSTM/Transformer for temporal modeling
* Add macro indicators and earnings transcripts
* Explore post-prediction portfolio optimization
* Blend neural and tree-based models (e.g., stacking)

---

Project by Sameer Rahman
