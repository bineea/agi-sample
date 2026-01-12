# -*- coding: utf-8 -*-
import argparse
import os
import re
import json
import time
import unicodedata
from typing import Any, Optional, Dict, List
import requests
import urllib3

LIST_URL = "https://ironman.lenovo.com/api/Resumes/Management"
DETAIL_URL = "https://ironman.lenovo.com/api/Resumes/Details"
ATTACH_URL = "https://ironman.lenovo.com/api/Resumes/Attachfile"


def normalize_filename(name: str) -> str:
    # 规范化并移除不合法字符
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r'[\\/:*?"<>|\r\n]+', "_", name).strip() or "file"


def find_key_recursively(obj: Any, target_key: str) -> Optional[Any]:
    # 在嵌套 JSON 中递归查找指定 key
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == target_key:
                return v
            found = find_key_recursively(v, target_key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for it in obj:
            found = find_key_recursively(it, target_key)
            if found is not None:
                return found
    return None


def parse_content_disposition(cd: Optional[str]) -> Optional[str]:
    # 解析 Content-Disposition 里的文件名
    if not cd:
        return None
    # filename*（RFC 5987）
    m = re.search(r"filename\*\s*=\s*[^']*'[^']*'([^;]+)", cd, flags=re.IGNORECASE)
    if m:
        return normalize_filename(m.group(1))
    # 普通 filename
    m = re.search(r'filename\s*=\s*"?([^";]+)"?', cd, flags=re.IGNORECASE)
    if m:
        return normalize_filename(m.group(1))
    return None


def get_list_page(session: requests.Session, status: str, page_index: int, page_size: int) -> List[Dict[str, Any]]:
    params = {
        "status": status,
        "pageIndex": page_index,
        "pageSize": page_size,
    }
    resp = session.get(LIST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    # 期望结构：{ data: { list: [ { data: { itemNo: "" } } ] } }
    items = (((data or {}).get("data") or {}).get("list") or [])
    return items


def get_attach_id_for_item(session: requests.Session, item_no: str) -> Optional[str]:
    url = f"{DETAIL_URL}?itemNo={item_no}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()

    # 先尝试 JSON，优先从已知的嵌套结构提取
    try:
        js = resp.json()
        print(f"js: {js}")

        # 优先尝试已知路径：data.consultantPhotoInfo.attachFileId
        if isinstance(js, dict):
            data = js.get("data")
            if isinstance(data, dict):
                consultant_photo_info = data.get("otherAttachmentInfo")
                if isinstance(consultant_photo_info, dict):
                    attach_id = consultant_photo_info.get("attachFileId")
                    if attach_id:
                        return str(attach_id)

        # 如果直接路径失败，使用递归查找作为备选
        attach_id = find_key_recursively(js, "attachFileId")
        if attach_id:
            return str(attach_id)
    except ValueError:
        pass  # 非 JSON，尝试从 HTML/文本中提取

    # 从文本中提取（HTML/JavaScript 等）
    text = resp.text or ""
    m = re.search(r'attachFileId["\']\s*:\s*["\']([A-Za-z0-9_\-.:/]+)["\']', text)
    if not m:
        # 再尝试 URL/参数形式
        m = re.search(r'attachFileId=([A-Za-z0-9_\-.:/]+)', text)
    if m:
        return m.group(1)
    return None

def download_attach(session: requests.Session, attach_id: str, out_dir: str,
                    filename_hint: Optional[str] = None) -> str:
    url = f"{ATTACH_URL}?attachFileId={attach_id}"
    with session.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        filename = parse_content_disposition(r.headers.get("Content-Disposition"))
        if not filename:
            # 用 itemNo/attachId 兜底
            base = filename_hint or attach_id
            filename = normalize_filename(f"{base}")
            # 简单按 content-type 猜扩展名（可选）
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "pdf" in ctype and not filename.lower().endswith(".pdf"):
                filename += ".pdf"
            elif "msword" in ctype and not filename.lower().endswith(".doc"):
                filename += ".doc"
            elif "officedocument.wordprocessingml" in ctype and not filename.lower().endswith(".docx"):
                filename += ".docx"

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        # 如重名则加序号
        base, ext = os.path.splitext(path)
        counter = 1
        while os.path.exists(path):
            path = f"{base}({counter}){ext}"
            counter += 1

        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return path


def main():
    parser = argparse.ArgumentParser(description="批量下载简历附件（自测用）")
    parser.add_argument("--token", help="Authorization: Bearer <token>",
                        default="")
    parser.add_argument("--status", default="", help="列表筛选的 status 参数")
    parser.add_argument("--page-size", type=int, default=50, help="每页数量")
    parser.add_argument("--out", default="downloads", help="下载输出目录")
    parser.add_argument("--delay", type=float, default=0.2, help="请求间隔秒，用于限速")
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {args.token}",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "resume-downloader/1.0",
    })
    # 禁用SSL验证以解决证书验证失败问题
    session.verify = False
    # 禁用SSL警告
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    page = 1
    total_downloaded = 0
    while True:
        print(f"[信息] 拉取列表：pageIndex={page}, pageSize={args.page_size}, status={args.status}")
        try:
            items = get_list_page(session, args.status, page, args.page_size)
        except requests.HTTPError as e:
            print(f"[错误] 列表接口 HTTP {e.response.status_code}: {e}")
            if e.response is not None and e.response.status_code in (401, 403):
                print("[提示] 认证失败，请检查/更新 Bearer Token。")
            break
        except Exception as e:
            print(f"[错误] 列表请求失败：{e}")
            break

        if not items:
            print("[信息] 已无更多数据，停止遍历。")
            break

        for idx, it in enumerate(items, 1):
            item_no = ((((it or {}).get("data")) or {}).get("itemNo")) or ""
            if not item_no:
                print(f"[警告] 第{idx}条缺少 itemNo，跳过。")
                continue

            print(f"[信息] 处理 itemNo={item_no} -> 获取详情以提取 attachFileId")
            try:
                attach_id = get_attach_id_for_item(session, item_no)
            except requests.HTTPError as e:
                print(f"[错误] 详情接口 HTTP {e.response.status_code}（itemNo={item_no}）：{e}")
                if e.response is not None and e.response.status_code in (401, 403):
                    print("[提示] 认证失败，请检查/更新 Bearer Token。")
                continue
            except Exception as e:
                print(f"[错误] 详情请求失败（itemNo={item_no}）：{e}")
                continue

            if not attach_id:
                print(f"[警告] 未找到 attachFileId（itemNo={item_no}），跳过。")
                # TODO 临时测试，后续需要删除
                break
                # continue

            print(f"[信息] 下载附件 attachFileId={attach_id}")
            try:
                saved = download_attach(session, attach_id, args.out, filename_hint=item_no)
                print(f"[成功] 已保存：{saved}")
                total_downloaded += 1
            except requests.HTTPError as e:
                print(f"[错误] 下载接口 HTTP {e.response.status_code}（attachFileId={attach_id}）：{e}")
            except Exception as e:
                print(f"[错误] 下载失败（attachFileId={attach_id}）：{e}")

            time.sleep(args.delay)

        page += 1
        time.sleep(args.delay)

    print(f"[总结] 完成，成功下载文件数：{total_downloaded}")


if __name__ == "__main__":
    main()
