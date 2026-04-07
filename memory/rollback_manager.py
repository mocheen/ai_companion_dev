"""
记忆回溯管理器
记录Agent对记忆数据的增删改操作，支持回滚到任意历史节点
"""

import os
import json
import uuid
import logging
import shutil
import threading
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from memory.memory_models import (
    ShortTermMessage, MediumTermMemory, LongTermMemory,
    EmotionType, DialogueType, LongTermMemoryType
)

logger = logging.getLogger(__name__)


@dataclass
class RollbackOperation:
    """单个操作的回溯记录"""
    step: int
    tool_name: str
    action: str  # "create" | "delete" | "merge" | "clean_short_term"
    target_type: str  # "medium_term" | "long_term" | "short_term"
    created_ids: List[str] = field(default_factory=list)
    deleted_data: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "tool_name": self.tool_name,
            "action": self.action,
            "target_type": self.target_type,
            "created_ids": self.created_ids,
            "deleted_data": self.deleted_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RollbackOperation':
        return cls(
            step=data["step"],
            tool_name=data["tool_name"],
            action=data["action"],
            target_type=data["target_type"],
            created_ids=data.get("created_ids", []),
            deleted_data=data.get("deleted_data", []),
        )


@dataclass
class RollbackSnapshot:
    """一次Agent执行产生的回溯快照"""
    snapshot_id: str
    parent_id: Optional[str]
    agent_type: str  # "archive" | "reorganize"
    created_at: str
    operations: List[RollbackOperation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "parent_id": self.parent_id,
            "agent_type": self.agent_type,
            "created_at": self.created_at,
            "operations": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RollbackSnapshot':
        return cls(
            snapshot_id=data["snapshot_id"],
            parent_id=data.get("parent_id"),
            agent_type=data["agent_type"],
            created_at=data["created_at"],
            operations=[RollbackOperation.from_dict(op) for op in data.get("operations", [])],
        )


class RollbackManager:
    """记忆回溯管理器，负责记录操作、保存快照、执行回滚"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._lock = threading.Lock()
        self._agent_running = False
        self._current_snapshot: Optional[RollbackSnapshot] = None

        rollback_config = config.get("rollback", {})
        self.enabled = rollback_config.get("enabled", True)
        self.max_snapshots = rollback_config.get("max_snapshots", 50)
        self.storage_dir = rollback_config.get("storage_dir", "data/rollback_snapshots")

        if self.enabled:
            os.makedirs(self.storage_dir, exist_ok=True)
            logger.info(f"回溯管理器初始化完成，存储目录: {self.storage_dir}")
        else:
            logger.info("回溯功能已禁用")

    def is_enabled(self) -> bool:
        return self.enabled

    def begin_snapshot(self, agent_type: str) -> Optional[str]:
        """开始一次新的回溯快照记录"""
        if not self.enabled:
            return None

        with self._lock:
            self._agent_running = True
            parent_id = self._get_latest_snapshot_id()
            self._current_snapshot = RollbackSnapshot(
                snapshot_id=str(uuid.uuid4()),
                parent_id=parent_id,
                agent_type=agent_type,
                created_at=datetime.now().isoformat(),
            )
            logger.info(f"开始回溯快照: {self._current_snapshot.snapshot_id}, agent_type={agent_type}")
            return self._current_snapshot.snapshot_id

    def record_create(self, tool_name: str, target_type: str, created_ids: List[str]):
        """记录创建操作"""
        if not self.enabled or not self._current_snapshot:
            return
        step = len(self._current_snapshot.operations) + 1
        self._current_snapshot.operations.append(RollbackOperation(
            step=step,
            tool_name=tool_name,
            action="create",
            target_type=target_type,
            created_ids=created_ids,
        ))
        logger.debug(f"记录创建操作: {tool_name}, target={target_type}, ids={created_ids}")

    def record_delete(self, tool_name: str, target_type: str, deleted_data: List[Dict[str, Any]]):
        """记录删除操作"""
        if not self.enabled or not self._current_snapshot:
            return
        step = len(self._current_snapshot.operations) + 1
        self._current_snapshot.operations.append(RollbackOperation(
            step=step,
            tool_name=tool_name,
            action="delete",
            target_type=target_type,
            deleted_data=deleted_data,
        ))
        logger.debug(f"记录删除操作: {tool_name}, target={target_type}, count={len(deleted_data)}")

    def record_merge(self, tool_name: str, target_type: str,
                     deleted_data: List[Dict[str, Any]], created_ids: List[str]):
        """记录合并操作"""
        if not self.enabled or not self._current_snapshot:
            return
        step = len(self._current_snapshot.operations) + 1
        self._current_snapshot.operations.append(RollbackOperation(
            step=step,
            tool_name=tool_name,
            action="merge",
            target_type=target_type,
            deleted_data=deleted_data,
            created_ids=created_ids,
        ))
        logger.debug(f"记录合并操作: {tool_name}, deleted={len(deleted_data)}, created={len(created_ids)}")

    def record_clean_short_term(self, deleted_messages: List[Dict[str, Any]]):
        """记录短期记忆清理操作"""
        if not self.enabled or not self._current_snapshot:
            return
        step = len(self._current_snapshot.operations) + 1
        self._current_snapshot.operations.append(RollbackOperation(
            step=step,
            tool_name="clean_archived_messages",
            action="clean_short_term",
            target_type="short_term",
            deleted_data=deleted_messages,
        ))
        logger.debug(f"记录短期记忆清理: count={len(deleted_messages)}")

    def save_snapshot(self) -> bool:
        """保存当前快照到文件"""
        if not self.enabled or not self._current_snapshot:
            return False

        with self._lock:
            try:
                snapshot = self._current_snapshot
                if not snapshot.operations:
                    logger.info("快照无操作记录，跳过保存")
                    self._agent_running = False
                    self._current_snapshot = None
                    return False

                ts = datetime.fromisoformat(snapshot.created_at)
                filename = f"rollback_{snapshot.agent_type}_{ts.strftime('%Y%m%d_%H%M%S')}_{snapshot.snapshot_id[:8]}.json"
                filepath = os.path.join(self.storage_dir, filename)

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)

                logger.info(f"回溯快照已保存: {filename}, 操作数={len(snapshot.operations)}")

                self._cleanup_old_snapshots()

                self._agent_running = False
                self._current_snapshot = None
                return True

            except Exception as e:
                logger.error(f"保存回溯快照失败: {e}", exc_info=True)
                self._agent_running = False
                self._current_snapshot = None
                return False

    def discard_snapshot(self):
        """丢弃当前快照（Agent执行失败时调用）"""
        if not self.enabled:
            return
        with self._lock:
            logger.info("丢弃当前回溯快照")
            self._agent_running = False
            self._current_snapshot = None

    def is_agent_running(self) -> bool:
        """检查是否有Agent正在运行"""
        with self._lock:
            return self._agent_running

    def get_all_snapshots(self) -> List[Dict[str, Any]]:
        """获取所有快照的摘要信息列表"""
        snapshots = []
        if not self.enabled:
            return snapshots

        try:
            for filename in sorted(os.listdir(self.storage_dir)):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    snapshots.append({
                        "snapshot_id": data["snapshot_id"],
                        "parent_id": data.get("parent_id"),
                        "agent_type": data["agent_type"],
                        "created_at": data["created_at"],
                        "operations_count": len(data.get("operations", [])),
                        "filename": filename,
                    })
                except Exception as e:
                    logger.warning(f"读取快照文件失败 {filename}: {e}")
        except Exception as e:
            logger.error(f"列出快照文件失败: {e}")

        snapshots.sort(key=lambda x: x["created_at"])
        return snapshots

    def get_snapshot_detail(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """获取指定快照的完整详情"""
        if not self.enabled:
            return None

        filepath = self._find_snapshot_file(snapshot_id)
        if not filepath:
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.error(f"读取快照详情失败: {e}")
            return None

    def rollback_to(self, snapshot_id: str, vector_store, redis_client,
                    redis_key_short_term: str, redis_key_flow_state: str) -> Dict[str, Any]:
        """回滚到指定快照节点"""
        if not self.enabled:
            return {"success": False, "message": "回溯功能已禁用"}

        with self._lock:
            if self._agent_running:
                return {"success": False, "message": "Agent正在运行中，无法执行回溯"}

            try:
                all_snapshots = self.get_all_snapshots()
                target_index = None
                for i, s in enumerate(all_snapshots):
                    if s["snapshot_id"] == snapshot_id:
                        target_index = i
                        break

                if target_index is None:
                    return {"success": False, "message": f"未找到快照: {snapshot_id}"}

                # 需要回滚的快照 = 目标之后的所有快照，逆序
                snapshots_to_rollback = list(reversed(all_snapshots[target_index + 1:]))

                if not snapshots_to_rollback:
                    return {"success": False, "message": "该节点已是最新状态，无需回溯"}

                logger.info(f"开始回溯到 {snapshot_id}，需回滚 {len(snapshots_to_rollback)} 个快照")

                rolled_back_ops = 0
                deleted_files = []

                for snapshot_summary in snapshots_to_rollback:
                    detail = self.get_snapshot_detail(snapshot_summary["snapshot_id"])
                    if not detail:
                        logger.warning(f"跳过无法读取的快照: {snapshot_summary['snapshot_id']}")
                        continue

                    snapshot = RollbackSnapshot.from_dict(detail)
                    # 逆序执行操作
                    for op in reversed(snapshot.operations):
                        self._rollback_operation(op, vector_store, redis_client,
                                                 redis_key_short_term, redis_key_flow_state)
                        rolled_back_ops += 1

                    # 删除已回滚的快照文件
                    filepath = self._find_snapshot_file(snapshot_summary["snapshot_id"])
                    if filepath:
                        os.remove(filepath)
                        deleted_files.append(os.path.basename(filepath))

                logger.info(f"回溯完成: 回滚了 {rolled_back_ops} 个操作，删除了 {len(deleted_files)} 个快照文件")

                return {
                    "success": True,
                    "message": f"已回溯到节点 {snapshot_id[:8]}，回滚了 {rolled_back_ops} 个操作",
                    "rolled_back_count": len(snapshots_to_rollback),
                    "operations_rolled_back": rolled_back_ops,
                    "deleted_files": deleted_files,
                }

            except Exception as e:
                logger.error(f"回溯执行失败: {e}", exc_info=True)
                return {"success": False, "message": f"回溯失败: {str(e)}"}

    def _rollback_operation(self, op: RollbackOperation, vector_store, redis_client,
                            redis_key_short_term: str, redis_key_flow_state: str):
        """执行单个操作的回滚"""
        if op.action == "create":
            # 创建操作 → 回滚时删除
            for mem_id in op.created_ids:
                if op.target_type == "medium_term":
                    vector_store.delete_medium_term_memory(mem_id)
                    logger.debug(f"回滚-create: 删除中期记忆 {mem_id}")
                elif op.target_type == "long_term":
                    vector_store.delete_long_term_memory(mem_id)
                    logger.debug(f"回滚-create: 删除长期记忆 {mem_id}")

        elif op.action == "delete":
            # 删除操作 → 回滚时恢复
            for item in op.deleted_data:
                self._restore_memory(vector_store, op.target_type, item)

        elif op.action == "merge":
            # 合并操作 → 删除创建的，恢复删除的
            for mem_id in op.created_ids:
                vector_store.delete_medium_term_memory(mem_id)
                logger.debug(f"回滚-merge: 删除合并后的记忆 {mem_id}")
            for item in op.deleted_data:
                self._restore_memory(vector_store, op.target_type, item)

        elif op.action == "clean_short_term":
            # 短期记忆清理 → 恢复被清理的消息
            self._restore_short_term_memory(redis_client, redis_key_short_term, op.deleted_data)

    def _restore_memory(self, vector_store, target_type: str, item: Dict[str, Any]):
        """恢复一条记忆到向量数据库"""
        mem_id = item.get("id")
        data = item.get("data")
        if not data:
            logger.warning(f"跳过无数据的记忆恢复: {mem_id}")
            return

        try:
            if target_type == "medium_term":
                memory = MediumTermMemory.from_dict(data)
                vector_store.add_medium_term_memory(memory)
                logger.debug(f"恢复中期记忆: {mem_id}")
            elif target_type == "long_term":
                memory = LongTermMemory.from_dict(data)
                vector_store.add_long_term_memory(memory)
                logger.debug(f"恢复长期记忆: {mem_id}")
        except Exception as e:
            logger.error(f"恢复记忆失败 {mem_id}: {e}")

    def _restore_short_term_memory(self, redis_client, redis_key: str,
                                   deleted_messages: List[Dict[str, Any]]):
        """恢复短期记忆到Redis"""
        try:
            data_json = redis_client.get(redis_key)
            current_data = json.loads(data_json) if data_json else []

            for msg_data in deleted_messages:
                current_data.append(msg_data)

            current_data.sort(key=lambda x: x.get("timestamp", ""))
            redis_client.set(redis_key, json.dumps(current_data, ensure_ascii=False))
            logger.info(f"恢复短期记忆: {len(deleted_messages)} 条")
        except Exception as e:
            logger.error(f"恢复短期记忆失败: {e}", exc_info=True)

    def _get_latest_snapshot_id(self) -> Optional[str]:
        """获取最新快照的ID作为parent_id"""
        snapshots = self.get_all_snapshots()
        if snapshots:
            return snapshots[-1]["snapshot_id"]
        return None

    def _find_snapshot_file(self, snapshot_id: str) -> Optional[str]:
        """根据snapshot_id查找文件路径"""
        try:
            for filename in os.listdir(self.storage_dir):
                if not filename.endswith(".json"):
                    continue
                filepath = os.path.join(self.storage_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("snapshot_id") == snapshot_id:
                        return filepath
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"查找快照文件失败: {e}")
        return None

    def _cleanup_old_snapshots(self):
        """清理超出上限的旧快照文件"""
        try:
            files = []
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    filepath = os.path.join(self.storage_dir, filename)
                    files.append((filepath, os.path.getmtime(filepath)))

            if len(files) <= self.max_snapshots:
                return

            files.sort(key=lambda x: x[1])
            to_delete = len(files) - self.max_snapshots

            for filepath, _ in files[:to_delete]:
                os.remove(filepath)
                logger.info(f"清理旧快照: {os.path.basename(filepath)}")

            logger.info(f"清理完成: 删除了 {to_delete} 个旧快照")
        except Exception as e:
            logger.error(f"清理旧快照失败: {e}")
