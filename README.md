# 🌲 Forest Cover Type Predictor

> Predicting which tree species dominates a wilderness area based on terrain and cartographic features — built for the U.S. Forest Service and environmental planning use cases.

---

## 📌 Business Problem / Motivation

Classifying forest cover type across large land areas is a critical task for environmental monitoring, conservation planning, and land management. Manually surveying vast forest regions is time-consuming, expensive, and difficult to scale.

This project builds a machine learning model that predicts the dominant tree cover type (out of 7 species) for any 30x30 meter section of forest land based solely on terrain and cartographic variables — helping organizations like the **U.S. Forest Service** make faster, more consistent, and scalable forest classification decisions.

---

## 📋 Project Overview

| | |
|---|---|
| **Goal** | Classify forest cover type into 1 of 7 tree species |
| **Approach** | Multi-class classification with SMOTE to handle class imbalance |
| **Best Model** | SMOTE + Random Forest (tuned with RandomizedSearchCV) |
| **Key Result** | **F1 Macro: 0.9256** · Accuracy: 0.9507 · Balanced Accuracy: 0.9362 |
| **Deployment** | Interactive Streamlit web app |

---

## 📦 Data

| | |
|---|---|
| **Source** | [Forest Cover Type Dataset — Kaggle / UCI ML Repository](https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset) |
| **Type** | Tabular, multi-class classification |
| **Size** | 581,012 rows × 55 columns |
| **Target** | `Cover_Type` — 7 forest cover classes |

### Cover Type Classes
| Class | Tree Type |
|---|---|
| 1 | Spruce / Fir |
| 2 | Lodgepole Pine |
| 3 | Ponderosa Pine |
| 4 | Cottonwood / Willow |
| 5 | Aspen |
| 6 | Douglas Fir |
| 7 | Krummholz |

### Key Features
- **Terrain:** Elevation, Aspect, Slope
- **Distances:** Horizontal/Vertical Distance to Hydrology, Roadways, Fire Points
- **Hillshade:** at 9am, Noon, 3pm
- **Categorical (one-hot encoded):** Wilderness Area (4 areas), Soil Type (40 types)

---

## 🧹 Data Preprocessing

1. **No missing values** or duplicate rows found — dataset was already clean
2. **Stratified 60/20/20 split** — preserves class distribution across train/val/test sets
   - Train: 348,607 rows | Validation: 116,203 rows | Test: 116,202 rows
3. **Median Imputation** — included in pipeline for robustness
4. **Standard Scaling** — applied to all numeric features via `StandardScaler`
5. **SMOTE** (Synthetic Minority Over-sampling Technique) — applied inside cross-validation folds only to prevent data leakage, generating synthetic examples of underrepresented cover types
6. All steps wrapped in a single `ImbPipeline` for reproducibility

---

## 📊 Exploratory Data Analysis (EDA)

Key visualizations generated during EDA:

- **Class Distribution Bar Chart** — Revealed extreme class imbalance (Classes 1 and 2 dominate; Classes 3–7 are minorities), motivating the use of SMOTE and Macro F1 as the primary metric
- **Correlation Heatmap** — Full 55×55 feature correlation map to detect multicollinearity
- **Tree Type Proportions by Wilderness Area** — Stacked bar chart directly answering the research question: different wilderness areas show strongly different cover type distributions
- **Feature Distribution by Cover Type** — Elevation, slope, and hillshade values vary significantly across species, confirming strong predictive signals

> 📁 See the `images/` folder for all exported visualizations.

---

## 🤖 Modeling Approach

We tested a progression of models, from simple baselines to advanced ensemble methods:

| Model | Type | Rationale |
|---|---|---|
| Logistic Regression | Baseline | Linear baseline; expected weak on non-linear relationships |
| Decision Tree | Baseline | Interpretable but prone to overfitting |
| Random Forest | Advanced | Ensemble of trees — stronger, more stable, handles tabular data well |
| XGBoost | Advanced | Gradient boosting — competitive on structured data |
| **SMOTE + Random Forest** | **Final** | **Best F1 Macro + handles class imbalance explicitly** |

Macro F1-score was chosen as the **primary evaluation metric** because the dataset is imbalanced — accuracy alone would favor majority classes and mask poor performance on minority classes. Macro F1 gives equal weight to all 7 classes.

---

## ⚙️ Model Training

- **Tool:** `scikit-learn`, `imbalanced-learn`, `RandomizedSearchCV`
- **Cross-validation:** `StratifiedKFold` (5 folds) — preserves class ratios in each fold
- **Tuning Method:** `RandomizedSearchCV` with 20 iterations (faster than grid search on 581K rows)

### Best Hyperparameters Found
| Parameter | Value |
|---|---|
| `smote__k_neighbors` | 3 |
| `clf__n_estimators` | 300 |
| `clf__max_features` | `sqrt` |
| `clf__max_depth` | None (fully grown trees) |
| `clf__min_samples_split` | 2 |
| `clf__min_samples_leaf` | 1 |

**Best Cross-Validation F1 Macro:** `0.9209`

---

## 📈 Results

### Final Test Set Performance (held-out, evaluated once)

| Metric | Score |
|---|---|
| **F1 Macro** | **0.9256** |
| **Accuracy** | **0.9507** |
| **Balanced Accuracy** | **0.9362** |

### Model Comparison Table

| Model | F1 Macro (Val) | Accuracy (Val) | Balanced Acc (Val) |
|---|---|---|---|
| Logistic Regression | — | — | — |
| Decision Tree | — | — | — |
| XGBoost | — | — | — |
| **SMOTE + Random Forest** | **0.9256** | **0.9507** | **0.9362** |

> 📁 See `results/` folder for the full comparison metrics and confusion matrix.

### Stability Check (5 Random Seeds)
| Metric | Value |
|---|---|
| Average F1 Macro | 0.9253 |
| Std Deviation | 0.0013 |

✅ Very low standard deviation confirms the model is **stable and not overfitting to a lucky split**.

---

## 🔍 Model Interpretation

We used two complementary explainability techniques to understand what drives predictions:

### 1. Permutation Feature Importance
Measures how much the model's F1 Macro drops when a feature's values are randomly shuffled — a model-agnostic importance measure that reflects true predictive contribution.

**Top Features:**
1. `Elevation` — by far the most important predictor
2. `Horizontal_Distance_To_Roadways`
3. `Horizontal_Distance_To_Fire_Points`
4. `Horizontal_Distance_To_Hydrology`
5. `Wilderness_Area` indicators

### 2. SHAP Values (SHapley Additive exPlanations)
Used `shap.TreeExplainer` on 500 test samples for three levels of explanation:
- **Beeswarm plot** — shows direction and magnitude of each feature's effect per class
- **Global bar plot** — mean |SHAP| across all classes confirms Elevation dominates
- **Waterfall plot** — local explanation of a single prediction, showing exactly how each feature pushed the prediction up or down

**Key Finding:** High elevation strongly predicts Spruce/Fir (Class 1) while low elevation and proximity to water strongly predicts Cottonwood/Willow (Class 4).

> 📁 See `images/` folder for all SHAP and feature importance plots.

---

## 💡 Key Insights

- **Elevation is the strongest predictor** — tree species occupy distinct elevation bands in the Rocky Mountains
- **Wilderness area matters** — Rawah and Neota are dominated by Spruce/Fir, while Cache la Poudre shows much more diversity
- **SMOTE significantly improved minority class coverage** — without it, Classes 4, 5, and 7 were frequently misclassified
- **The model is production-ready** — F1 Macro of 0.9256 with a std of 0.0013 across 5 seeds shows strong generalization
- **Practical impact:** This model can reduce manual forest survey time and support scalable environmental planning decisions

---

## 🏁 Conclusion

We built a Forest Cover Type Predictor that classifies 7 tree species across 581K land sections using terrain and cartographic features. The final model — SMOTE + Random Forest with tuned hyperparameters — achieved a Macro F1 of **0.9256** on the held-out test set, outperforming all baselines while remaining stable and interpretable.

The Streamlit web app makes the model accessible to non-technical users, allowing real-time predictions by adjusting terrain sliders — turning a machine learning model into a practical decision-support tool.

---

## 🔮 Future Work

- **Add more models:** LightGBM, CatBoost for potential further improvement
- **Feature engineering:** Derived features like distance ratios or elevation bands
- **Class-specific tuning:** Per-class threshold optimization for even better minority class recall
- **Larger SHAP sample:** Full test set SHAP for more reliable global explanation
- **Model retraining pipeline:** Automate retraining when new survey data becomes available
- **Mobile-friendly app:** Improve Streamlit UI for field use on tablets/phones

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/forest-cover-predictor.git
cd forest-cover-predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
> The app will automatically download `best_model.joblib` from Google Drive on first launch.

### 4. Run the Notebook (optional — to retrain)
Open `notebooks/Code_-_C2_FINAL__best_model_only_.ipynb` and run all cells in order:
1. Load dataset (downloads via `kagglehub`)
2. EDA → Preprocessing → Model Training → Evaluation → SHAP
3. Run the last cell to save `best_model.joblib`

---

## 📁 Repository Structure

```
forest-cover-predictor/
│
├── README.md                                  # Project documentation (this file)
├── requirements.txt                           # All Python dependencies
├── runtime.txt                                # Python version (python-3.10)
├── app.py                                     # Streamlit web app for predictions
│
├── notebooks/
│   ├── Code_-_C2_FINAL__best_model_only_.ipynb     # Final model: SMOTE + Random Forest
│   ├── Code_-_C2_first_presentation.ipynb          # Capstone 1 / first presentation notebook
│   └── Code_-_C2_second_presentation.ipynb         # Second presentation notebook
│
├── data/
│   └── README.md                              # Dataset info + Kaggle download instructions
│
├── models/
│   └── README.md                              # Model info + Google Drive download link
│
├── results/
│   └── model_comparison.csv                   # Metrics for all models compared
│
└── images/
    ├── class_distribution.png                 # Cover type class imbalance bar chart
    ├── correlation_heatmap.png                # Feature correlation heatmap
    ├── wilderness_area_proportions.png        # Tree types by wilderness area
    ├── confusion_matrix_val.png               # Validation confusion matrix
    ├── confusion_matrix_test.png              # Final test confusion matrix
    ├── permutation_importance.png             # Top 20 permutation feature importances
    ├── shap_beeswarm.png                      # SHAP beeswarm plot (Class 1)
    ├── shap_global_bar.png                    # SHAP global mean |SHAP| bar chart
    └── shap_waterfall.png                     # SHAP waterfall for single prediction
```

---

## 📦 Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---|---|
| streamlit | >=1.32.0 |
| pandas | 2.2.2 |
| numpy | 2.0.2 |
| scikit-learn | 1.6.1 |
| imbalanced-learn | 0.14.1 |
| xgboost | 3.2.0 |
| joblib | >=1.2.0 |
| gdown | >=4.7.1 |

---
