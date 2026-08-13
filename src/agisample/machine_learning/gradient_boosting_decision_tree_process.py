import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import difflib
from sklearn.metrics import precision_recall_curve, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


class GradientBoostingDecisionTreeProcess:
    def __init__(self, model_type='xgboost'):
        """
        初始化付款匹配模型
        model_type: 'xgboost' 或 'lightgbm'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def create_features(self, payment, order):
        """
        为单个付款-订单对创建特征
        """
        features = {}

        # 1. 金额相关特征
        features['amount_diff'] = abs(payment['amount'] - order['amount'])
        features['amount_ratio'] = min(payment['amount'], order['amount']) / max(payment['amount'], order['amount'])
        features['amount_diff_percentage'] = abs(payment['amount'] - order['amount']) / payment['amount']

        # 2. 日期相关特征
        payment_date = pd.to_datetime(payment['payment_date'])
        order_date = pd.to_datetime(order['order_date'])
        date_diff = abs((payment_date - order_date).days)
        features['date_diff'] = date_diff
        features['date_diff_weeks'] = date_diff / 7
        features['is_same_month'] = int(payment_date.month == order_date.month)
        features['is_same_week'] = int(payment_date.isocalendar()[1] == order_date.isocalendar()[1])

        # 3. 文本相似度特征
        # 3.1 付款人-客户名称相似度
        features['name_exact_match'] = int(str(payment['payer']).lower() == str(order['customer']).lower())
        features['name_similarity'] = difflib.SequenceMatcher(
            None,
            str(payment['payer']).lower(),
            str(order['customer']).lower()
        ).ratio()

        # 3.2 描述相似度
        features['desc_exact_match'] = int(
            str(payment['description']).lower() == str(order['order_description']).lower())
        features['desc_similarity'] = difflib.SequenceMatcher(
            None,
            str(payment['description']).lower(),
            str(order['order_description']).lower()
        ).ratio()

        # 4. 业务规则特征
        features['amount_match'] = int(abs(payment['amount'] - order['amount']) < 0.01)
        features['date_within_30_days'] = int(date_diff <= 30)

        return features

    def prepare_dataset(self, payments_df, orders_df):
        """
        准备训练数据集
        """
        features_list = []
        labels = []
        pairs = []  # 存储付款-订单对的索引

        for payment_idx, payment in payments_df.iterrows():
            for order_idx, order in orders_df.iterrows():
                # 创建特征
                features = self.create_features(payment, order)
                features_list.append(features)

                # 创建标签（1表示匹配，0表示不匹配）
                label = 1 if payment['order_id'] == order['order_id'] else 0
                labels.append(label)

                # 存储索引对
                pairs.append((payment_idx, order_idx))

        # 转换为DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()

        return features_df, np.array(labels), pairs

    def train(self, payments_df, orders_df):
        """
        训练模型
        """
        # 准备数据
        features_df, labels, _ = self.prepare_dataset(payments_df, orders_df)

        # 分割数据集
        X_train, X_val, y_train, y_val = train_test_split(
            features_df, labels, test_size=0.2, random_state=42
        )

        # 根据选择的模型类型训练
        if self.model_type == 'xgboost':
            self.train_xgboost(X_train, X_val, y_train, y_val)
        else:  # lightgbm
            self.train_lightgbm(X_train, X_val, y_train, y_val)

        # 特征重要性分析
        self.plot_feature_importance()

        # 返回验证集评估结果
        return self.evaluate(X_val, y_val)

    def train_xgboost(self, X_train, X_val, y_train, y_val):
        """
        训练XGBoost模型
        """
        # 创建数据集
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)

        # 设置参数
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 6,
            'eta': 0.1,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)  # 处理类别不平衡
        }

        # 训练模型
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=100
        )

    def train_lightgbm(self, X_train, X_val, y_train, y_val):
        """
        训练LightGBM模型
        """
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=self.feature_names)

        # 设置参数
        params = {
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'max_depth': 6,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': sum(y_train == 0) / sum(y_train == 1)
        }

        # 训练模型
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            early_stopping_rounds=50,
            verbose_eval=100
        )

    def predict(self, payment, orders_df, threshold=0.5):
        """
        预测单个付款凭证的匹配订单
        """
        features_list = []
        for _, order in orders_df.iterrows():
            features = self.create_features(payment, order)
            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        # 根据模型类型进行预测
        if self.model_type == 'xgboost':
            dtest = xgb.DMatrix(features_df, feature_names=self.feature_names)
            probas = self.model.predict(dtest)
        else:  # lightgbm
            probas = self.model.predict(features_df)

        # 找到最佳匹配
        best_match_idx = np.argmax(probas)
        best_match_prob = probas[best_match_idx]

        return {
            'match_index': best_match_idx,
            'confidence': best_match_prob,
            'is_match': best_match_prob >= threshold,
            'all_probas': probas  # 返回所有订单的匹配概率
        }

    def evaluate(self, X_val, y_val):
        """
        评估模型性能
        """
        if self.model_type == 'xgboost':
            dval = xgb.DMatrix(X_val, feature_names=self.feature_names)
            probas = self.model.predict(dval)
        else:  # lightgbm
            probas = self.model.predict(X_val)

        # 计算不同阈值下的precision和recall
        precisions, recalls, thresholds = precision_recall_curve(y_val, probas)

        # 计算F1分数
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]

        # 使用最佳阈值的预测结果
        predictions = (probas >= best_threshold).astype(int)
        report = classification_report(y_val, predictions, output_dict=True)

        return {
            'best_threshold': best_threshold,
            'best_f1': max(f1_scores),
            'classification_report': report
        }

    def plot_feature_importance(self):
        """
        绘制特征重要性图
        """
        plt.figure(figsize=(12, 6))
        if self.model_type == 'xgboost':
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.get_score(importance_type='gain')
            })
        else:  # lightgbm
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importance(importance_type='gain')
            })

        importance_df = importance_df.sort_values('importance', ascending=True)

        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Feature Importance (gain)')
        plt.title('Feature Importance Analysis')
        plt.tight_layout()
        plt.show()




# # 准备示例数据
# payments_df = pd.DataFrame({
#     'payment_id': ['P1', 'P2'],
#     'amount': [1000, 2000],
#     'payment_date': ['2024-01-01', '2024-01-02'],
#     'payer': ['公司A', '公司B'],
#     'description': ['产品购买', '服务费用'],
#     'order_id': ['ORDER001', 'ORDER002']  # 用于训练
# })
#
# orders_df = pd.DataFrame({
#     'order_id': ['ORDER001', 'ORDER002'],
#     'amount': [1000, 2000],
#     'order_date': ['2024-01-01', '2024-01-02'],
#     'customer': ['公司A', '公司B'],
#     'order_description': ['产品购买', '服务费用']
# })
#
# # 训练XGBoost模型
# xgb_matcher = GBDTPaymentMatcher(model_type='xgboost')
# eval_results = xgb_matcher.train(payments_df, orders_df)
# print("XGBoost评估结果:", eval_results)
#
# # 训练LightGBM模型
# lgb_matcher = GBDTPaymentMatcher(model_type='lightgbm')
# eval_results = lgb_matcher.train(payments_df, orders_df)
# print("LightGBM评估结果:", eval_results)
#
# # 预测新的付款凭证
# new_payment = {
#     'amount': 1050,
#     'payment_date': '2024-01-01',
#     'payer': '公司A',
#     'description': '产品采购'
# }
#
# result = xgb_matcher.predict(new_payment, orders_df)
# print("预测结果:", result)