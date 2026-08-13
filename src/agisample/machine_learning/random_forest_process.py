import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from fuzzywuzzy import fuzz


class RandomForestProcess:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def _create_features(self, payment_data, order_data):
        """生成匹配特征"""
        features = []
        labels = []

        for _, payment in payment_data.iterrows():
            payment_features = []
            for _, order in order_data.iterrows():
                # 1. 金额差异特征
                amount_diff = abs(payment['amount'] - order['amount'])
                amount_ratio = min(payment['amount'], order['amount']) / max(payment['amount'], order['amount'])

                # 2. 时间差异特征（假设有 payment_date 和 order_date 字段）
                days_diff = abs((payment['payment_date'] - order['order_date']).days)

                # 3. 文本相似度特征
                # 付款人与订单客户名称的模糊匹配分数
                name_similarity = fuzz.ratio(str(payment['payer_name']), str(order['customer_name'])) / 100.0

                # 付款备注与订单号的模糊匹配分数
                reference_similarity = fuzz.partial_ratio(str(payment['reference']), str(order['order_number'])) / 100.0

                # 4. 组合所有特征
                features.append([
                    amount_diff,
                    amount_ratio,
                    days_diff,
                    name_similarity,
                    reference_similarity
                ])

                # 5. 添加标签（示例中使用简单规则生成训练标签）
                is_match = (amount_diff < 0.01 and
                            days_diff < 7 and
                            name_similarity > 0.8 and
                            reference_similarity > 0.7)
                labels.append(1 if is_match else 0)

        return np.array(features), np.array(labels)

    def train(self, payment_data, order_data):
        """训练匹配模型"""
        # 生成训练特征
        X, y = self._create_features(payment_data, order_data)

        # 特征标准化
        X_scaled = self.scaler.fit_transform(X)

        # 训练模型
        self.model.fit(X_scaled, y)

        # 输出训练报告
        y_pred = self.model.predict(X_scaled)
        print("训练集表现:")
        print(classification_report(y, y_pred))

    def predict(self, payment_data, order_data, threshold=0.5):
        """预测匹配结果"""
        # 生成预测特征
        X, _ = self._create_features(payment_data, order_data)
        X_scaled = self.scaler.transform(X)

        # 获取匹配概率
        probas = self.model.predict_proba(X_scaled)[:, 1]

        # 根据阈值筛选匹配结果
        matches = []
        current_idx = 0

        for i, payment in payment_data.iterrows():
            payment_matches = []
            for j, order in order_data.iterrows():
                if probas[current_idx] >= threshold:
                    payment_matches.append({
                        'payment_id': payment.name,
                        'order_id': order.name,
                        'confidence': probas[current_idx]
                    })
                current_idx += 1

            # 对当前支付记录的匹配结果按置信度排序
            payment_matches.sort(key=lambda x: x['confidence'], reverse=True)
            matches.extend(payment_matches)

        return pd.DataFrame(matches)


# 使用示例：
"""
# 准备示例数据
payment_data = pd.DataFrame({
    'payment_date': pd.date_range('2024-01-01', '2024-01-10'),
    'amount': [1000, 1500, 2000],
    'payer_name': ['公司A', '公司B', '公司C'],
    'reference': ['ORDER001', 'ORDER002', 'ORDER003']
})

order_data = pd.DataFrame({
    'order_date': pd.date_range('2024-01-01', '2024-01-10'),
    'amount': [1000, 1500, 2000],
    'customer_name': ['公司A', '公司B', '公司C'],
    'order_number': ['ORDER001', 'ORDER002', 'ORDER003']
})

# 初始化匹配器
matcher = PaymentMatcher()

# 训练模型
matcher.train(payment_data, order_data)

# 预测匹配
matches = matcher.predict(payment_data, order_data, threshold=0.8)
print("匹配结果:")
print(matches)
"""
