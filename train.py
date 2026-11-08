"""Train five classifiers from the local CSV. Run: python train.py"""
from pathlib import Path
import json, warnings
import joblib, pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"/"online_shoppers_intention.csv"
MD=ROOT/"model"; 
MD.mkdir(exist_ok=True)
TARGET="Revenue"; 
TEST_SIZE=0.20; 
RANDOM_STATE=42

# Change these hyperparameters to experiment, then rerun python train.py.
MODEL_SETTINGS={
 "Logistic Regression":{"C":1.0,"max_iter":2000,"class_weight":"balanced","solver":"lbfgs"},
 "Decision Tree":{"criterion":"gini","max_depth":12,"min_samples_split":20,"min_samples_leaf":5,"class_weight":"balanced"},
 "K-Nearest Neighbors":{"n_neighbors":7,"weights":"distance","metric":"minkowski"},
 "Gaussian Naive Bayes":{"var_smoothing":1e-9},
 "Support Vector Machine":{"C":2.0,"kernel":"rbf","gamma":"scale","probability":True,"class_weight":"balanced"}
}
FILES={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib","K-Nearest Neighbors":"knn.joblib","Gaussian Naive Bayes":"gaussian_nb.joblib","Support Vector Machine":"svm.joblib"}

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Put the local CSV here: {DATA}")
    df=pd.read_csv(DATA); df.columns=df.columns.astype(str).str.strip()
    
    if TARGET not in df: raise ValueError(f"Missing target column {TARGET}")
    
    df=df.dropna(how="all").dropna(axis=1,how="all").copy()

    y=df[TARGET].astype(str).str.strip().str.lower().map({"false":0,"true":1,"0":0,"1":1,"no":0,"yes":1})
    
    if y.isna().any(): raise ValueError("Revenue must contain True/False or 0/1.")
    
    y=y.astype(int)
    X=df.drop(columns=[TARGET])

    #Categorical
    cats=["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend"]
    absent=[c for c in cats if c not in X]
    if absent: raise ValueError(f"Missing expected categorical columns: {absent}")

    #Numerical
    nums=[c for c in X if c not in cats]
    for c in cats: X[c]=X[c].astype("string").fillna("Missing")
    for c in nums: X[c]=pd.to_numeric(X[c],errors="coerce")

    if len(X)<500 or X.shape[1]<12: raise ValueError("Dataset must have at least 500 rows and 12 input features.")

    #Pipleine
    prep=ColumnTransformer([
      ("numeric",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),nums),
      ("categorical",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cats)
    ],remainder="drop",verbose_feature_names_out=False)

    #Train Test Split
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)

    s=MODEL_SETTINGS
    models={
      "Logistic Regression":LogisticRegression(**s["Logistic Regression"],random_state=RANDOM_STATE),
      "Decision Tree":DecisionTreeClassifier(**s["Decision Tree"],random_state=RANDOM_STATE),
      "K-Nearest Neighbors":KNeighborsClassifier(**s["K-Nearest Neighbors"]),
      "Gaussian Naive Bayes":GaussianNB(**s["Gaussian Naive Bayes"]),
      "Support Vector Machine":SVC(**s["Support Vector Machine"],random_state=RANDOM_STATE)}
    
    rows=[]; 
    fitted={}
    
    for name,clf in models.items():
        print("Training",name)
        pipe=Pipeline([("preprocessor",prep),("classifier",clf)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); pipe.fit(Xtr,ytr)
        pred=pipe.predict(Xte); prob=pipe.predict_proba(Xte)[:,1]
        rows.append({"Model":name,"Accuracy":accuracy_score(yte,pred),"AUC":roc_auc_score(yte,prob),
          "Precision":precision_score(yte,pred,zero_division=0),"Recall":recall_score(yte,pred,zero_division=0),
          "F1-score":f1_score(yte,pred,zero_division=0),"MCC":matthews_corrcoef(yte,pred)})
        joblib.dump(pipe,MD/FILES[name]); fitted[name]=pipe
    test=Xte.copy(); test[TARGET]=yte.map({0:"False",1:"True"}); test.to_csv(ROOT/"test_data.csv",index=False)
    pd.DataFrame(rows).to_csv(ROOT/"model_metrics.csv",index=False)
    counts=y.value_counts()
    feature_info=[]
    for c in X:
        cat=c in cats
        feature_info.append({"Feature":c,"Role":"Input feature","Type":"Categorical" if cat else "Numerical",
          "Data type":"Category/text" if cat else "Numeric","Unique values":int(X[c].nunique()),
          "Missing values":int(df[c].isna().sum())})
    meta={"dataset_name":"Online Shoppers Purchasing Intention","source":"UCI Machine Learning Repository","uci_id":468,
      "rows":len(df),"input_features":X.shape[1],"categorical_feature_count":len(cats),"numerical_feature_count":len(nums),
      "target":TARGET,"target_type":"Binary classification","positive_class":"True (purchase)","negative_class":"False (no purchase)",
      "class_counts":{"No purchase (False)":int(counts.get(0,0)),"Purchase (True)":int(counts.get(1,0))},
      "feature_info":feature_info,"feature_columns":list(X.columns),"training_rows":len(Xtr),"test_rows":len(Xte),
      "test_size":TEST_SIZE,"random_state":RANDOM_STATE,"model_settings":s,
      "decision_tree":{"implementation":"scikit-learn DecisionTreeClassifier (CART-style classification tree)",**s["Decision Tree"]}}
    (MD/"metadata.json").write_text(json.dumps(meta,indent=2))
    tree=fitted["Decision Tree"].named_steps["classifier"]
    names=fitted["Decision Tree"].named_steps["preprocessor"].get_feature_names_out()
    (ROOT/"decision_tree_rules.txt").write_text(export_text(tree,feature_names=list(names)))
    plt.figure(figsize=(24,12))
    plot_tree(tree,feature_names=list(names),class_names=["No purchase","Purchase"],filled=True,rounded=True,max_depth=3,fontsize=7)
    plt.tight_layout(); plt.savefig(ROOT/"decision_tree_visualisation.png",dpi=160,bbox_inches="tight"); plt.close()
    pd.DataFrame({"Feature":names,"Importance":tree.feature_importances_}).sort_values("Importance",ascending=False).to_csv(ROOT/"decision_tree_feature_importance.csv",index=False)
    print("Training complete. Saved five model pipelines, test_data.csv, metrics and tree explainability files.")
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=="__main__": main()
