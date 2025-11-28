import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def load_and_preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df.columns = df.columns.str.strip()

    print("="*70)
    print("DATA OVERVIEW")
    print("="*70)
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in ["education", "self_employed"] if c in df.columns]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    label_encoders = {}
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip().replace({'nan': None})
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_val)
        cats = pd.Categorical(df[col]).categories.tolist()
        mapping = {cat: code for code, cat in enumerate(cats)}
        inverse_mapping = {v: k for k, v in mapping.items()}
        default_code = mapping.get(mode_val, 0)
        df[col] = df[col].map(mapping).fillna(default_code).astype(int)
        label_encoders[col] = {'mapping': mapping, 'inverse_mapping': inverse_mapping, 'default': default_code}

    if 'loan_status' in df.columns:
        df['loan_status'] = df['loan_status'].astype(str).str.strip()
        le = LabelEncoder()
        df['loan_status'] = le.fit_transform(df['loan_status'])
        label_encoders['loan_status'] = le

    if 'loan_amount' in df.columns and 'income_annum' in df.columns:
        df["loan_to_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1)
    else:
        df["loan_to_income_ratio"] = 0

    asset_cols = [c for c in ['residential_assets_value', 'commercial_assets_value', 'luxury_assets_value', 'bank_asset_value'] if c in df.columns]
    if asset_cols:
        df['total_assets'] = df[asset_cols].sum(axis=1)
    else:
        df['total_assets'] = 0

    if 'loan_amount' in df.columns and 'total_assets' in df.columns:
        df['asset_to_loan_ratio'] = df['total_assets'] / (df['loan_amount'] + 1)
    else:
        df['asset_to_loan_ratio'] = 0

    if 'loan_amount' in df.columns and 'loan_term' in df.columns:
        df['debt_burden'] = df['loan_amount'] / (df['loan_term'] + 1)
    else:
        df['debt_burden'] = 0

    return df, label_encoders

def get_hyperparameter_grids():
    """Define hyperparameter search spaces for each model"""
    param_grids = {
        "Random Forest": {
            'clf__n_estimators': randint(50, 300),
            'clf__max_depth': [None, 5, 10, 15, 20],
            'clf__min_samples_split': randint(2, 11),
            'clf__min_samples_leaf': randint(1, 5),
            'clf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
        },
        "Gradient Boosting": {
            'clf__n_estimators': randint(50, 200),
            'clf__learning_rate': uniform(0.01, 0.19),
            'clf__max_depth': randint(3, 8),
            'clf__min_samples_split': randint(2, 11),
            'clf__min_samples_leaf': randint(1, 5)
        },
        "Extra Trees": {
            'clf__n_estimators': randint(50, 300),
            'clf__max_depth': [None, 5, 10, 15, 20],
            'clf__min_samples_split': randint(2, 11),
            'clf__min_samples_leaf': randint(1, 5),
            'clf__max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7]
        },
        "Logistic Regression": {
            'clf__C': uniform(0.01, 9.99),
            'clf__penalty': ['l2'],
            'clf__solver': ['liblinear', 'saga']
        }
    }
    return param_grids

def create_pipeline(model):
    """Create a pipeline with preprocessing and the given model"""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

def hyperparameter_tuning(X_train, y_train, n_iter=50, cv_folds=5):
    """Perform hyperparameter tuning for selected models"""
    
    models_to_tune = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42)
    }
    
    param_grids = get_hyperparameter_grids()
    tuned_results = {}
    
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING")
    print("="*70)
    
    for name, model in models_to_tune.items():
        print(f"\nTuning {name}...")
        
        pipe = create_pipeline(model)
        
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grids[name],
            n_iter=n_iter,
            cv=cv_folds,
            scoring='roc_auc',
            n_jobs=1,
            random_state=42,
            verbose=1
        )
        
        try:
            search.fit(X_train, y_train)
        except Exception as e:
            print(f"  Error during hyperparameter tuning for {name}: {e}")
            print(f"  Falling back to default parameters for {name}")
            pipe.fit(X_train, y_train)
            tuned_results[name] = {
                "model": pipe,
                "cv_score": cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring='roc_auc').mean(),
                "best_params": "Default (tuning failed)",
                "search_object": None
            }
            continue
        
        print(f"  Best CV Score: {search.best_score_:.4f}")
        print(f"  Best Parameters: {search.best_params_}")
        
        tuned_results[name] = {
            "model": search.best_estimator_,
            "cv_score": search.best_score_,
            "best_params": search.best_params_,
            "search_object": search
        }
    
    return tuned_results

def train_baseline_models(X_train, y_train, cv_folds=5):
    """Train baseline models without tuning for comparison"""
    
    baseline_models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }
    
    baseline_results = {}
    
    print("\n" + "="*70)
    print("BASELINE MODELS (NO TUNING)")
    print("="*70)
    
    for name, model in baseline_models.items():
        print(f"\nTraining {name}...")
        
        pipe = create_pipeline(model)
        
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        
        pipe.fit(X_train, y_train)
        
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        baseline_results[name] = {
            "model": pipe,
            "cv_score": cv_scores.mean(),
            "cv_std": cv_scores.std()
        }
    
    return baseline_results

def evaluate_all_models(tuned_results, baseline_results, X_test, y_test):
    """Evaluate all models and select the best one"""
    
    all_results = {}
    
    print("\n" + "="*70)
    print("FINAL MODEL EVALUATION")
    print("="*70)
    
    for name, info in tuned_results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        print(f"\n{name} (Tuned):")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV ROC AUC: {info['cv_score']:.4f}")
        if roc_auc:
            print(f"  Test ROC AUC: {roc_auc:.4f}")
        print(f"  Best Parameters: {info['best_params']}")
        
        all_results[f"{name} (Tuned)"] = {
            "model": model,
            "accuracy": accuracy,
            "cv_score": info['cv_score'],
            "roc_auc": roc_auc,
            "is_tuned": True
        }
    
    for name, info in baseline_results.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_proba)
            else:
                roc_auc = None
        except:
            roc_auc = None
        
        print(f"\n{name} (Baseline):")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  CV ROC AUC: {info['cv_score']:.4f}")
        if roc_auc:
            print(f"  Test ROC AUC: {roc_auc:.4f}")
        
        all_results[f"{name} (Baseline)"] = {
            "model": model,
            "accuracy": accuracy,
            "cv_score": info['cv_score'],
            "roc_auc": roc_auc,
            "is_tuned": False
        }
    
    return all_results

def plot_model_comparison(all_results):
    """Create comprehensive model comparison plots"""
    
    models = []
    test_acc = []
    cv_auc = []
    test_auc = []
    is_tuned = []
    
    for name, info in all_results.items():
        models.append(name.replace(' (Tuned)', '\n(Tuned)').replace(' (Baseline)', '\n(Baseline)'))
        test_acc.append(info['accuracy'])
        cv_auc.append(info['cv_score'])
        test_auc.append(info.get('roc_auc', 0) if info.get('roc_auc') is not None else 0)
        is_tuned.append(info['is_tuned'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    colors = ['#FF6B6B' if tuned else '#4ECDC4' for tuned in is_tuned]
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(range(len(models)), test_acc, color=colors)
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.85, 1.02)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels(models, rotation=45, ha='right')
    
    for bar, acc in zip(bars1, test_acc):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax2 = axes[0, 1]
    bars2 = ax2.bar(range(len(models)), cv_auc, color=colors)
    ax2.set_title('Cross-Validation ROC AUC', fontweight='bold')
    ax2.set_ylabel('ROC AUC')
    ax2.set_ylim(0.85, 1.02)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=45, ha='right')
    
    for bar, auc in zip(bars2, cv_auc):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax3 = axes[1, 0]
    valid_test_auc = [auc for auc in test_auc if auc > 0]
    valid_models = [model for i, model in enumerate(models) if test_auc[i] > 0]
    valid_colors = [colors[i] for i, auc in enumerate(test_auc) if auc > 0]
    
    if valid_test_auc:
        bars3 = ax3.bar(range(len(valid_models)), valid_test_auc, color=valid_colors)
        ax3.set_title('Test ROC AUC', fontweight='bold')
        ax3.set_ylabel('ROC AUC')
        ax3.set_ylim(0.85, 1.02)
        ax3.set_xticks(range(len(valid_models)))
        ax3.set_xticklabels(valid_models, rotation=45, ha='right')
        
        for bar, auc in zip(bars3, valid_test_auc):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No ROC AUC data available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Test ROC AUC', fontweight='bold')
    
    ax4 = axes[1, 1]
    tuned_acc = [acc for i, acc in enumerate(test_acc) if is_tuned[i]]
    baseline_acc = [acc for i, acc in enumerate(test_acc) if not is_tuned[i]]
    
    box_data = []
    labels = []
    if baseline_acc:
        box_data.append(baseline_acc)
        labels.append('Baseline')
    if tuned_acc:
        box_data.append(tuned_acc)
        labels.append('Hypertuned')
    
    if box_data:
        bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
        colors_box = ['#4ECDC4', '#FF6B6B']
        for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax4.set_title('Tuned vs Baseline Distribution', fontweight='bold')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_ylim(0.85, 1.02)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#FF6B6B', label='Hypertuned'),
                      Patch(facecolor='#4ECDC4', label='Baseline')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nModel comparison plot saved as 'model_comparison.png'")

def plot_hyperparameter_analysis(tuned_results):
    """Plot hyperparameter analysis for tuned models"""
    
    if not tuned_results:
        print("No hyperparameter tuning results to plot")
        return
    
    n_models = len(tuned_results)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
    
    ax1 = axes[0, 0]
    models = list(tuned_results.keys())
    cv_scores = [results['cv_score'] for results in tuned_results.values()]
    
    bars = ax1.bar(models, cv_scores, color=plt.cm.Set3(range(len(models))))
    ax1.set_title('Best CV ROC AUC Scores', fontweight='bold')
    ax1.set_ylabel('ROC AUC Score')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, score in zip(bars, cv_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10)
    
    ax2 = axes[0, 1]
    tree_models = {}
    for name, results in tuned_results.items():
        params = results['best_params']
        if 'clf__n_estimators' in params:
            tree_models[name] = params['clf__n_estimators']
    
    if tree_models:
        bars2 = ax2.bar(tree_models.keys(), tree_models.values(), 
                        color=plt.cm.Set2(range(len(tree_models))))
        ax2.set_title('Optimal N_Estimators', fontweight='bold')
        ax2.set_ylabel('Number of Estimators')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, n_est in zip(bars2, tree_models.values()):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{n_est}', ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No n_estimators data', ha='center', va='center',
                transform=ax2.transAxes)
        ax2.set_title('Optimal N_Estimators', fontweight='bold')
    
    ax3 = axes[1, 0]
    depth_models = {}
    for name, results in tuned_results.items():
        params = results['best_params']
        if 'clf__max_depth' in params:
            depth = params['clf__max_depth']
            depth_models[name] = depth if depth is not None else 0
    
    if depth_models:
        bars3 = ax3.bar(depth_models.keys(), depth_models.values(),
                        color=plt.cm.Set1(range(len(depth_models))))
        ax3.set_title('Optimal Max Depth', fontweight='bold')
        ax3.set_ylabel('Max Depth')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, depth in zip(bars3, depth_models.values()):
            height = bar.get_height()
            label = 'None' if depth == 0 else str(depth)
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    label, ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'No max_depth data', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Optimal Max Depth', fontweight='bold')
    
    ax4 = axes[1, 1]
    lr_models = {}
    for name, results in tuned_results.items():
        params = results['best_params']
        if 'clf__learning_rate' in params:
            lr_models[name] = params['clf__learning_rate']
    
    if lr_models:
        bars4 = ax4.bar(lr_models.keys(), lr_models.values(),
                        color=plt.cm.Pastel1(range(len(lr_models))))
        ax4.set_title('Optimal Learning Rate', fontweight='bold')
        ax4.set_ylabel('Learning Rate')
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, lr in zip(bars4, lr_models.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                    f'{lr:.3f}', ha='center', va='bottom', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'No learning rate data', ha='center', va='center',
                transform=ax4.transAxes)
        ax4.set_title('Optimal Learning Rate', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    plt.savefig('hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Hyperparameter analysis plot saved as 'hyperparameter_analysis.png'")

def plot_confusion_matrix_comparison(all_results, X_test, y_test, label_encoders):
    """Plot confusion matrices for top models"""
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['cv_score'], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Confusion Matrix Comparison - Top 4 Models', fontsize=16, fontweight='bold')
    axes = axes.ravel()
    
    for i, (name, info) in enumerate(sorted_results):
        model = info['model']
        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        
        if hasattr(label_encoders['loan_status'], 'classes_'):
            labels = label_encoders['loan_status'].classes_
        else:
            labels = ['Class 0', 'Class 1']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(f'{name}\nAccuracy: {info["accuracy"]:.3f}', fontweight='bold')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()
    
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrices plot saved as 'confusion_matrices.png'")

def select_best_model(all_results):
    """Select the best model based on CV ROC AUC score"""
    
    comparison_data = []
    for name, info in all_results.items():
        comparison_data.append([
            name,
            info['accuracy'],
            info['cv_score'],
            info.get('roc_auc', 'N/A'),
            'Yes' if info['is_tuned'] else 'No'
        ])
    
    df_comparison = pd.DataFrame(comparison_data, 
                               columns=['Model', 'Test Accuracy', 'CV ROC AUC', 'Test ROC AUC', 'Tuned'])
    df_comparison = df_comparison.sort_values('CV ROC AUC', ascending=False)
    
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    print(df_comparison.to_string(index=False))
    
    best_model_name = df_comparison.iloc[0]['Model']
    best_model = all_results[best_model_name]['model']
    
    print(f"\n" + "="*70)
    print(f"BEST MODEL: {best_model_name}")
    print(f"CV ROC AUC: {all_results[best_model_name]['cv_score']:.4f}")
    print(f"Test Accuracy: {all_results[best_model_name]['accuracy']:.4f}")
    print("="*70)
    
    return best_model, best_model_name, all_results[best_model_name]

def get_user_input_and_predict(model, label_encoders, feature_cols):
    print("\nEnter details for prediction:")

    try:
        no_of_dependents = int(input("Dependents: "))
        education_raw = input("Education (Graduate/Not Graduate): ").strip()
        self_employed_raw = input("Self Employed (Yes/No): ").strip()
        income_annum = float(input("Annual Income: "))
        loan_amount = float(input("Loan Amount: "))
        loan_term = int(input("Loan Term: "))
        cibil_score = int(input("CIBIL Score: "))
        residential_assets_value = float(input("Residential Assets Value: "))
        commercial_assets_value = float(input("Commercial Assets Value: "))
        luxury_assets_value = float(input("Luxury Assets Value: "))
        bank_asset_value = float(input("Bank Asset Value: "))
    except:
        print("Invalid input.")
        return

    if isinstance(label_encoders.get('education'), dict):
        edu_map = label_encoders['education']['mapping']
        edu_def = label_encoders['education']['default']
        education = edu_map.get(education_raw, edu_def)
    else:
        education = label_encoders['education'].transform([education_raw])[0]

    if isinstance(label_encoders.get('self_employed'), dict):
        se_map = label_encoders['self_employed']['mapping']
        se_def = label_encoders['self_employed']['default']
        self_emp = se_map.get(self_employed_raw, se_def)
    else:
        self_emp = label_encoders['self_employed'].transform([self_employed_raw])[0]

    data = {
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_emp,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }

    input_df = pd.DataFrame([data])

    input_df["loan_to_income_ratio"] = loan_amount / (income_annum + 1)
    input_df["total_assets"] = (residential_assets_value +
                                commercial_assets_value +
                                luxury_assets_value +
                                bank_asset_value)
    input_df["asset_to_loan_ratio"] = input_df["total_assets"] / (loan_amount + 1)
    input_df["debt_burden"] = loan_amount / (loan_term + 1)

    input_df = input_df[feature_cols]

    pred = model.predict(input_df)[0]
    
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_df)[0]
        prob_approved = proba[1] if len(proba) > 1 else proba[0]
        print(f"Prediction Probability: {prob_approved:.4f}")

    result = label_encoders["loan_status"].inverse_transform([pred])[0]
    print(f"\nLoan Prediction: {result}")

def interactive_prediction_loop(model, label_encoders, feature_cols):
    """Interactive prediction loop that asks for multiple predictions"""
    while True:
        print("\n" + "="*50)
        try:
            get_user_input_and_predict(model, label_encoders, feature_cols)
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error during prediction: {e}")
        
        print("\n" + "-"*50)
        while True:
            another = input("Would you like to make another prediction? (y/n): ").strip().lower()
            if another in ['y', 'yes', '1']:
                break
            elif another in ['n', 'no', '0']:
                print("Thank you for using the loan prediction system!")
                return
            else:
                print("Please enter 'y' for yes or 'n' for no.")

def main():
    print("Bank Loan Approval Prediction with Hyperparameter Tuning")
    print("=" * 60)
    
    df, label_encoders = load_and_preprocess_data("loan_approval_dataset.csv")

    feature_cols = [c for c in df.columns if c not in ["loan_id", "loan_status"]]
    X = df[feature_cols]
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")

    tuned_results = hyperparameter_tuning(X_train, y_train, n_iter=30, cv_folds=5)
    
    baseline_results = train_baseline_models(X_train, y_train, cv_folds=5)
    
    all_results = evaluate_all_models(tuned_results, baseline_results, X_test, y_test)
    
    print("\n" + "="*70)
    print("GENERATING PERFORMANCE VISUALIZATIONS")
    print("="*70)
    
    plot_model_comparison(all_results)
    plot_hyperparameter_analysis(tuned_results)
    plot_confusion_matrix_comparison(all_results, X_test, y_test, label_encoders)
    
    best_model, best_model_name, best_info = select_best_model(all_results)

    print("\n" + "="*70)
    print(f"DETAILED EVALUATION: {best_model_name}")
    print("="*70)
    
    y_pred = best_model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n" + "="*70)
    print("READY FOR INTERACTIVE PREDICTION")
    print("="*70)
    interactive_prediction_loop(best_model, label_encoders, feature_cols)

    return best_model, all_results, label_encoders, feature_cols

if __name__ == "__main__":
    main()