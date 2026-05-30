# Medicine-Engine-Demo-Introduction


# 高血压风险预测系统 - 基于多模型异质集成
### 高血压预测网址直达:
https://medicine-engine-demo-introductiongit-nrq2w2ljgy72w3wlv9edf9.streamlit.app/
### 模型训练数据集:
https://www.kaggle.com/datasets/zkskhurram/blood-pressure-by-age-global-dataset

---

## 项目简介
本项目构建了一套完整的机器学习预测框架，包含8种模型对比（单一树、三融合、四融合），并通过SHAP可解释性分析验证了特征工程的有效性。最终四模型融合在测试集上取得 **AUC=0.7900**，过拟合差距仅 **0.0394**（全模型最小）。

---

## 核心亮点
- **白名单防泄露**：剔除收缩压/舒张压等诊断指标，AUC从0.99回归至0.79真实水平
- **8模型对比**：XGBoost/LightGBM/CatBoost/随机森林/逻辑回归/SVM + 三融合 + 四融合
- **异质集成**：树模型家族 + 逻辑回归，线性与非线性互补
- **SHAP可解释性**：Age_BMI交互特征贡献度第一，验证特征工程有效性
- **过拟合控制**：四融合过拟合差距仅0.0394，全模型最小
- **全栈部署**：Streamlit + 五个Tab功能闭环，README前端化
- **工程结构清晰**：方法与主体分离，`class BaseDataHandler`处理数据传入和特征列，`class RandomForestForFeature`处理特征，`class HypertensionModel`进行特征交互工程和特征工程，并评估参数将其打印在各个模型的评估日志上，采用类定义`class 模型名(HypertensionModel)`模型方便修改和main调用

---

## 📁 核心代码
```
PART1.嵌入随机森林模型筛选特征与特征交互工程(from model_training.py)
#=============================================================
#随机森林模型
#=============================================================
class RandomForestForFeatures(BaseDataHandler):
    #===============筛选模型训练===============
    def model_training(self,x_train, y_train):
        self.featuremodel = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            oob_score = True
        )
        self.featuremodel.fit(x_train, y_train)
        self.feature_importances_ = self.featuremodel.feature_importances_
        print(f"筛选模型 OOB Score: {self.featuremodel.oob_score_:.4f}")
        return self.featuremodel
    #===============遍历并得到新特征列===============
    def select_top_features(self, top_k=10):
        if not hasattr(self, 'feature_importances_'):
            raise ValueError("请先调用 model_training()训练筛选模型")
        indices = np.argsort(self.feature_importances_)[::-1]
        self.selected_features = [self.feature_cols[i] for i in indices[:top_k]]
        return self.selected_features
class HypertensionModel(RandomForestForFeatures):
...
#===============特征交互===============
    def create_interaction_features(self, df, selected_features):
        df = df.copy()
        # ===============只对筛选后存在的特征构造交互===============
        if 'Age' in selected_features and 'BMI' in selected_features:
            df['Age_BMI'] = df['Age'] * df['BMI']
        if 'Age' in selected_features and 'Family_Hx_Hypertension_Yes' in selected_features:
            df['Age_FamilyHx'] = df['Age'] * df['Family_Hx_Hypertension_Yes']
        if 'BMI' in selected_features and 'Diabetes_Yes' in selected_features:
            df['BMI_Diabetes'] = df['BMI'] * df['Diabetes_Yes']
        if 'Smoking_Status' in selected_features and 'Alcohol_Use' in selected_features:
            df['Smoking_Alcohol'] = df['Smoking_Status'] * df['Alcohol_Use']
        # ===============更新特征列列表===============
        original_cols = set(selected_features + [self.label])
        new_interaction_cols = [col for col in df.columns if col not in original_cols]
        enhanced_features = selected_features + new_interaction_cols
        return df, enhanced_features
...
PART2.消融实验(from model_training.py)
#=============================================================
#三融合模型
#=============================================================
from sklearn.linear_model import LogisticRegression

class Fusion_Model_demo(HypertensionModel):
    # ===============声明名称并传入evaluation中===============
    def __init__(self):
        super().__init__(model_name="Fusion_Model_demo")
        self.xgb_model = None
        self.cat_model = None
        self.rf_model = None
        self.meta_model = None

    def demo_model_training(self, x_train, x_val, y_train, y_val):
        # ===============XGBoost模型训练===============
        # ===============正负样本比例计算===============
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        # ===============模型参数配置===============
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.3,
            reg_lambda=1.0,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            early_stopping_rounds=30,
            eval_metric='auc'
        )
        eval_set = [(x_val, y_val)]
        self.xgb_model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        # ==========模型CatBoost==========
        self.cat_model = cat.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_seed=42,
            logging_level='Silent',
            allow_writing_files=False
        )
        self.cat_model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                           early_stopping_rounds=20, verbose=False)
        # ===============随机森林模型训练===============
        self.rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(x_train, y_train)
        # ========== 元模型：逻辑回归学习融合权重 ==========
        xgb_val = self.xgb_model.predict_proba(x_val)[:, 1]
        cat_val = self.cat_model.predict_proba(x_val)[:, 1]
        rf_val = self.rf_model.predict_proba(x_val)[:, 1]
        x_meta_val = np.column_stack([xgb_val, cat_val, rf_val])
        # ==========元模型XGBoost==========
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=42
        )
        self.meta_model.fit(x_meta_val, y_val)
        # ========== 验证集评估 ==========
        meta_val_proba = self.meta_model.predict_proba(x_meta_val)[:, 1]
        val_auc = roc_auc_score(y_val, meta_val_proba)
        # ==========打印元模型特征重要性（看哪个基模型贡献大）==========
        print(f"[三模型融合] 元模型特征重要性: XGB={self.meta_model.feature_importances_[0]:.3f}, "
              f"Cat={self.meta_model.feature_importances_[1]:.3f}, "
              f"RF={self.meta_model.feature_importances_[2]:.3f} ")
        print(f"[三模型融合] 验证集 AUC: {val_auc:.4f}")
        # ========== 保存模型 ==========
        self.model = {
            'xgb_model': self.xgb_model,
            'cat_model': self.cat_model,
            'rf_model': self.rf_model,
            'meta_model': self.meta_model
        }
    #==========保存参数为HypertensionModel中参数打印调用==========
    def predict_proba(self, x):
        xgb_proba = self.xgb_model.predict_proba(x)[:, 1]
        cat_proba = self.cat_model.predict_proba(x)[:, 1]
        rf_proba = self.rf_model.predict_proba(x)[:, 1]
        X_meta = np.column_stack([xgb_proba, cat_proba, rf_proba])
        return self.meta_model.predict_proba(X_meta)
    def predict(self, x, threshold=0.5):
        proba = self.predict_proba(x)[:, 1]
        return (proba >= threshold).astype(int)
#=============================================================
#四融合模型
#=============================================================
class Fusion_Model(HypertensionModel):
    # ===============声明名称并传入evaluation中===============
    def __init__(self):
        super().__init__(model_name="Fusion_Model")
        self.xgb_model = None
        self.cat_model = None
        self.rf_model = None
        self.lr_model = None
        self.meta_model = None
    def model_training(self, x_train, x_val, y_train, y_val):
        #===============XGBoost模型训练===============
        # ===============正负样本比例计算===============
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count
        # ===============模型参数配置===============
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            reg_alpha=0.3,
            reg_lambda=1.0,
            max_depth=4,
            subsample=0.7,
            colsample_bytree=0.7,
            early_stopping_rounds=30,
            eval_metric='auc'
        )
        eval_set = [(x_val, y_val)]
        self.xgb_model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            verbose=True
        )
        # ==========模型CatBoost==========
        self.cat_model = cat.CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3,
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_seed=42,
            logging_level='Silent',
            allow_writing_files=False
        )
        self.cat_model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                           early_stopping_rounds=20, verbose=False)
        #===============随机森林模型训练===============
        self.rf_model = RandomForestClassifier(
            n_estimators=500,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(x_train, y_train)
        #===============逻辑回归===============
        self.lr_model = LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.lr_model.fit(x_train, y_train)
        # ========== 元模型：逻辑回归学习融合权重 ==========
        xgb_val = self.xgb_model.predict_proba(x_val)[:, 1]
        cat_val = self.cat_model.predict_proba(x_val)[:, 1]
        rf_val = self.rf_model.predict_proba(x_val)[:, 1]
        lr_val = self.lr_model.predict_proba(x_val)[:, 1]

        x_meta_val = np.column_stack([xgb_val, cat_val, rf_val, lr_val])

        # ==========元模型XGBoost==========
        self.meta_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            scale_pos_weight=scale_pos_weight,
            eval_metric='auc',
            random_state=42
        )
        self.meta_model.fit(x_meta_val, y_val)

        # ========== 验证集评估 ==========
        meta_val_proba = self.meta_model.predict_proba(x_meta_val)[:, 1]
        val_auc = roc_auc_score(y_val, meta_val_proba)

        # 打印元模型特征重要性（看哪个基模型贡献大）
        print(f"[四模型融合] 元模型特征重要性: XGB={self.meta_model.feature_importances_[0]:.3f}, "
              f"Cat={self.meta_model.feature_importances_[1]:.3f}, "
              f"RF={self.meta_model.feature_importances_[2]:.3f}, "
              f"LR={self.meta_model.feature_importances_[3]:.3f}")
        print(f"[四模型融合] 验证集 AUC: {val_auc:.4f}")

        # ========== 保存模型 ==========
        self.model = {
            'xgb_model': self.xgb_model,
            'cat_model': self.cat_model,
            'rf_model': self.rf_model,
            'lr_model': self.lr_model,
            'meta_model': self.meta_model
        }
        import pickle
        with open('meta_model.pkl', 'wb') as f:
            pickle.dump(self.meta_model, f)
    # ==========保存参数为HypertensionModel中参数打印调用==========
    def predict_proba(self, x):
        xgb_proba = self.xgb_model.predict_proba(x)[:, 1]
        cat_proba = self.cat_model.predict_proba(x)[:, 1]
        rf_proba = self.rf_model.predict_proba(x)[:, 1]
        lr_proba = self.lr_model.predict_proba(x)[:, 1]
        x_meta = np.column_stack([xgb_proba, cat_proba, rf_proba, lr_proba])
        return self.meta_model.predict_proba(x_meta)
    def predict(self, x, threshold=0.5):
        proba = self.predict_proba(x)[:, 1]
        return (proba >= threshold).astype(int)
```
---

## 开始运行

### 配置环境所需要的安装包
```
pip install requirements.txt
```

### 运行Streamlit应用
```
cd D:\Hypertension Project\src  
D:
streamlit run app.py
```
---
## C++移植工程(ONNX Runtime与Catboost C API)
python架构存在一定的局限性，无法高效部署于本地，于是嵌入onnx runtime与catboost c api进行模型移植，实现离线运行预测功能。

### 依赖库
```
https://github.com/microsoft/onnxruntime/releases/tag/v1.26.0
https://github.com/catboost/catboost/releases
```

---

### 模型性能汇总

| 模型 | 测试AUC | F1分数 | 过拟合差距 |
|:---|:---:|:---:|:---:|
| XGBoost | 0.7862 | 0.7806 | 0.0897 |
| LightGBM | 0.7833 | 0.7844 | 0.0891 |
| CatBoost | 0.7909 | 0.7798 | 0.0542 |
| RandomForest | 0.7814 | 0.7790 | 0.0638 |
| SVM | 0.7304 | 0.7864 | 0.0203 |
| Logistic | 0.7976 | 0.7570 | 0.0059 |
| 三融合 | 0.7831 | 0.7843 | 0.0536 |
| **四融合** | **0.7900** | **0.7821** | **0.0394** |

## 技术栈
- **语言**：Python 3.8+ / C++
- **机器学习**：XGBoost / LightGBM / CatBoost / Scikit-learn
- **可视化**：Matplotlib / Seaborn / SHAP
- **前端**：Streamlit
- **数据处理**：Pandas / NumPy
- **模型持久化**：Joblib
- **移植**: ONNX Runtime / CatBoost C API

## 最后
与统计建模大赛相关的内容皆由作者一人完成，其余队友参与内容为0

