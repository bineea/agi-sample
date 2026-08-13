from talon import quotations

# 示例邮件内容
email_text = """
发件人: 张三 <zhangsan@example.com>
收件人: 李四 <lisi@example.com>
发送时间: 2025-04-09 10:00
主题: 关于项目的初步讨论

您好，
这是我们项目的最新讨论内容……

-----原始邮件-----
发件人: 王五 <wangwu@example.com>
收件人: 张三 <zhangsan@example.com>
发送时间: 2025-04-08 15:30
主题: 项目初步讨论

尊敬的张先生，
关于项目的相关问题请参阅附件……

-----原始邮件-----
发件人: 李四 <lisi@example.com>
收件人: 王五 <wangwu@example.com>
发送时间: 2025-04-07 09:20
主题: 会议安排

各位好，
请确认会议时间……
"""

# 提取原始邮件内容（不包含引用部分）
extracted_content = quotations.extract_from_plain(email_text)
print("提取的内容:")
print(extracted_content)

# 如果您需要获取完整的内容，包括回复链，可以使用：
# extracted_with_quotations = quotations.extract_from_plain(email_text, keep_quotations=True)

# 分析回复链（获取不同层级的回复）
def analyze_thread(email_text):
    # 首先提取最新的回复（不含引用）
    latest_reply = quotations.extract_from_plain(email_text)

    # 保留引用部分，并尝试分析
    full_text = quotations.extract_from_plain(email_text)

    # 简单的分割方法（根据实际邮件格式可能需要调整）
    # 这里假设引用部分以 ">" 开头
    replies = []
    current_reply = ""

    for line in full_text.split('\n'):
        if line.startswith('>'):
            if current_reply:
                replies.append(current_reply.strip())
                current_reply = ""
            replies.append(line.strip())
        else:
            current_reply += line + "\n"

    if current_reply:
        replies.append(current_reply.strip())

    return {
        "latest_reply": latest_reply,
        "reply_chain": replies
    }


# 使用分析函数
result = analyze_thread(email_text)
print("最新回复:", result["latest_reply"])
print("\n回复链:")
for i, reply in enumerate(result["reply_chain"]):
    print(f"--- 回复 {i + 1} ---")
    print(reply)
    print()