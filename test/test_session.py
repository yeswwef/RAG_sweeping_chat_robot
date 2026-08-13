# -*- coding: utf-8 -*-
"""
test/test_session.py - 会话 CRUD 单元测试
运行：python -m pytest test/test_session.py -v
"""
import os
import sys
import tempfile

# 确保能 import 项目模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 用临时数据库，避免污染真实数据
import utils.db as db
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
db.DB_PATH = _tmp.name
db._conn = None  # 重置单例连接


def setup_function():
    db._conn = None
    db.get_conn()  # 初始化建表


def teardown_function():
    if db._conn:
        db._conn.close()
        db._conn = None


def test_create_and_count():
    sid1 = db.create_conversation()
    sid2 = db.create_conversation("第二个会话")
    assert db.count_conversations() == 2
    assert sid1 != sid2
    assert db.get_conversation(sid1)["title"] == "新对话"
    assert db.get_conversation(sid2)["title"] == "第二个会话"


def test_add_and_list_messages():
    sid = db.create_conversation()
    db.add_message(sid, "user", "你好")
    db.add_message(sid, "assistant", "您好！")
    db.add_message(sid, "user", "天气如何")
    msgs = db.list_messages(sid)
    assert len(msgs) == 3
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "你好"
    assert msgs[2]["content"] == "天气如何"
    # 会话间互不干扰
    sid2 = db.create_conversation()
    assert db.list_messages(sid2) == []


def test_update_time_and_title():
    sid = db.create_conversation("旧标题")
    db.update_conversation_time(sid, title="新标题")
    assert db.get_conversation(sid)["title"] == "新标题"
    assert db.get_conversation(sid)["updated_at"] >= db.get_conversation(sid)["created_at"]


def test_delete_cascades_messages():
    sid = db.create_conversation()
    db.add_message(sid, "user", "会被删除")
    db.add_message(sid, "assistant", "也会被删除")
    db.delete_conversation(sid)
    assert db.get_conversation(sid) is None
    assert db.list_messages(sid) == []
    assert db.count_conversations() == 0


def test_make_title():
    assert db.make_title("扫地机器人怎么保养？") == "扫地机器人怎么保养？"
    assert db.make_title("很长" * 30) == "很长" * 10  # 截断到20字
    assert db.make_title("   ") == "新对话"