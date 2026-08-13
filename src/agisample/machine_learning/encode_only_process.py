import torch
from torch import nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class EncodeOnlyProcess(Dataset):
    def __init__(self, payments, orders, tokenizer, max_length=128):
        """
        参数:
        payments: DataFrame, 包含付款凭证信息
        orders: DataFrame, 包含订单信息
        tokenizer: BERT tokenizer
        max_length: 序列最大长度
        """
        self.payments = payments
        self.orders = orders
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 生成正负样本对
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        """生成训练对，包括匹配和不匹配的样本"""
        positive_pairs = []  # 匹配的样本对
        negative_pairs = []  # 不匹配的样本对

        # 这里假设payments和orders中有一个match_id字段表示真实的匹配关系
        for idx, payment in self.payments.iterrows():
            # 添加正样本
            matching_order = self.orders[self.orders['match_id'] == payment['match_id']].iloc[0]
            positive_pairs.append({
                'payment': payment['text'],
                'order': matching_order['text'],
                'label': 1
            })

            # 添加负样本
            non_matching_orders = self.orders[self.orders['match_id'] != payment['match_id']].sample(1).iloc[0]
            negative_pairs.append({
                'payment': payment['text'],
                'order': non_matching_orders['text'],
                'label': 0
            })

        return positive_pairs + negative_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # 对文本进行编码
        encoded_pair = self.tokenizer(
            pair['payment'],
            pair['order'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'token_type_ids': encoded_pair['token_type_ids'].squeeze(),
            'label': torch.tensor(pair['label'], dtype=torch.long)
        }


class PaymentOrderMatcher(nn.Module):
    def __init__(self, bert_model_name='bert-base-chinese'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        # 使用[CLS]标记的输出作为整个序列的表示
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def train_model(model, train_loader, valid_loader, device, num_epochs=3):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, token_type_ids)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch + 1}:')
        print(f'Training Loss: {total_loss / len(train_loader):.4f}')
        print(f'Validation Loss: {valid_loss / len(valid_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')


def predict_match(model, tokenizer, payment_text, order_text, device, threshold=0.5):
    """预测两条记录是否匹配"""
    model.eval()

    # 编码输入
    encoded = tokenizer(
        payment_text,
        order_text,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    token_type_ids = encoded['token_type_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask, token_type_ids)
        probabilities = torch.softmax(outputs, dim=1)
        match_probability = probabilities[:, 1].item()

    return match_probability > threshold, match_probability


# 使用示例
def main():
    # 1. 准备数据
    # 这里需要替换成实际的数据加载逻辑
    payments_df = pd.DataFrame({
        'text': ['付款凭证1', '付款凭证2'],
        'match_id': [1, 2]
    })
    orders_df = pd.DataFrame({
        'text': ['订单1', '订单2'],
        'match_id': [1, 2]
    })

    # 2. 初始化tokenizer和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = PaymentOrderMatcher()

    # 3. 创建数据集和数据加载器
    dataset = PaymentOrderDataset(payments_df, orders_df, tokenizer)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=16)

    # 4. 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_model(model, train_loader, valid_loader, device)

    # 5. 预测示例
    payment_text = "付款金额10000元，收款方张三，日期2024-01-01"
    order_text = "订单金额10000元，客户张三，下单日期2024-01-01"
    is_match, probability = predict_match(model, tokenizer, payment_text, order_text, device)
    print(f"匹配结果: {'匹配' if is_match else '不匹配'}")
    print(f"匹配概率: {probability:.4f}")


if __name__ == "__main__":
    main()
