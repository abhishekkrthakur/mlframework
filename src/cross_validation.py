import pandas as pd
from sklearn import model_selection
import math

"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""


class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle, 
            problem_type="binary_classification",
            multilabel_delimiter=",",
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle,
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            
            num_fold_size=math.floor(len(self.dataframe)/self.num_folds)
            num_remainder =len(self.dataframe)%self.num_folds

            self.dataframe.sort_values(by=self.target_cols , inplace=True)
            df_remainder = self.dataframe.sample(n=num_remainder, random_state=42)
            df_kfold=self.dataframe[~self.dataframe.index.isin(df_remainder.index)]
            df_remainder=df_remainder.reset_index(drop=True)
            df_kfold=df_kfold.reset_index(drop=True)

            
            df_all=[]
            cnt=0
            for i in range(num_fold_size):
                df_subset=df_kfold.loc[i*self.num_folds:(i+1)*(self.num_folds)-1,:]
                df_subset = df_subset.sample(frac=1).reset_index(drop=True)
                cnt=cnt+1
                for j in range(self.num_folds):
                    if cnt==1:
                        df_all.append(df_subset.loc[[j],:])
                    else:
                        df_all[j]=df_all[j].append(df_subset.loc[[j],:], ignore_index=True)

            assigned_kfold=0
            if len(df_remainder)>0:
                for i in range(len(df_remainder)):
                    df_all[assigned_kfold]=df_all[assigned_kfold].append(df_remainder.loc[i,:],ignore_index=True)
                    assigned_kfold=assigned_kfold+1
                    if assigned_kfold>self.num_folds:
                        assigned_kfold=0
            
            for i in range(self.num_folds):
                if i==0:
                    df_all[i]['kfold']=i
                    self.dataframe=df_all[i]
                else:
                    df_all[i]['kfold']=i
                    self.dataframe=self.dataframe.append(df_all[i],ignore_index=True)

        
        
        elif self.problem_type.startswith("holdout_"):
            holdout_percentage = int(self.problem_type.split("_")[1])
            num_holdout_samples = int(len(self.dataframe) * holdout_percentage / 100)
            self.dataframe.loc[:len(self.dataframe) - num_holdout_samples, "kfold"] = 0
            self.dataframe.loc[len(self.dataframe) - num_holdout_samples:, "kfold"] = 1

        elif self.problem_type == "multilabel_classification":
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            targets = self.dataframe[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
            kf = model_selection.StratifiedKFold(n_splits=self.num_folds)
            for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=targets)):
                self.dataframe.loc[val_idx, 'kfold'] = fold

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("../input/train_reg.csv")
    cv = CrossValidation(df, shuffle=True, target_cols=["SalePrice"], 
                         problem_type="single_col_regression", num_folds=3, multilabel_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())
