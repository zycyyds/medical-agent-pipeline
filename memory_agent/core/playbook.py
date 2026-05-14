from __future__ import annotations

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from .models import (
    Bullet,
    SECTION_ORDER,
    SECTION_PREFIX,
    SECTION_TITLES,
    normalize_section,
)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+\-/\.]+|[\u4e00-\u9fff]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "of",
    "and",
    "or",
    "for",
    "with",
    "on",
    "in",
    "是",
    "的",
    "了",
    "和",
    "或",
    "与",
    "在",
    "对",
    "按",
    "将",
    "需要",
    "必须",
    "可以",
    "进行",
    "如果",
    "时",
    "后",
    "中",
    "把",
    "并",
}


def normalize_bullet_content(content: Any) -> str:
    return re.sub(r"\s+", " ", str(content or "")).strip()


def tokenize(text: str) -> set[str]:
    tokens = {item.lower() for item in _TOKEN_RE.findall(str(text or ""))}
    return {item for item in tokens if item and item not in _STOPWORDS}


def lexical_overlap_score(query_text: str, bullet_text: str) -> float:
    q = tokenize(query_text)
    b = tokenize(bullet_text)
    if not q or not b:
        return 0.0
    overlap = len(q & b)
    if overlap == 0:
        return 0.0
    return overlap / max(1.0, len(q))


def semantic_similarity(a: str, b: str) -> float:
    na = normalize_bullet_content(a).lower()
    nb = normalize_bullet_content(b).lower()
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


class ACEPlaybookManager:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.created_at = datetime.now().isoformat()
        self.version = "2.0"
        self.statistics: dict[str, Any] = {}
        self.bullets: dict[str, Bullet] = {}
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            self._ensure_statistics_defaults()
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            self._ensure_statistics_defaults()
            return

        self.created_at = raw.get("created_at") or self.created_at
        self.version = str(raw.get("version") or self.version)
        self.statistics = dict(raw.get("statistics") or {})
        self.bullets = {}
        for item in raw.get("entries", []):
            if not isinstance(item, dict):
                continue
            bullet = Bullet(
                id=str(item.get("id") or ""),
                content=str(item.get("content") or ""),
                section=item.get("section") or item.get("type") or "general",
                helpful_count=int(item.get("helpful_count") or 0),
                harmful_count=int(item.get("harmful_count") or 0),
                created_at=str(item.get("created_at") or ""),
                last_used=str(item.get("last_used") or ""),
                metadata=dict(item.get("metadata") or {}),
                status=str(item.get("status") or "active"),
            )
            if bullet.id:
                self.bullets[bullet.id] = bullet

        self._ensure_statistics_defaults()

    def _ensure_statistics_defaults(self) -> dict[str, Any]:
        stats = self.statistics
        stats.setdefault("total_executions", 0)
        stats.setdefault("unique_strategies", 0)
        stats.setdefault("last_reflection", None)
        stats.setdefault("total_bullets", 0)
        stats.setdefault("last_refine_at", None)
        stats.setdefault("duplicate_merges", 0)
        stats.setdefault("last_query", None)
        return stats

    def save(self, record_execution: bool = False) -> bool:
        stats = self._ensure_statistics_defaults()
        stats["total_bullets"] = len(self.bullets)
        stats["unique_strategies"] = len({
            normalize_bullet_content(item.content).lower()
            for item in self.bullets.values()
            if normalize_bullet_content(item.content)
        })
        if record_execution:
            stats["total_executions"] = int(stats.get("total_executions", 0)) + 1
            stats["last_reflection"] = datetime.now().isoformat()

        payload = {
            "version": self.version,
            "created_at": self.created_at,
            "entries": [item.to_dict() for item in sorted(self.bullets.values(), key=lambda x: x.id)],
            "statistics": stats,
        }
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def _generate_bullet_id(self, section: str) -> str:
        prefix = SECTION_PREFIX.get(section, "ctx")
        max_idx = 0
        for bullet_id in self.bullets:
            if not bullet_id.startswith(f"{prefix}-"):
                continue
            try:
                max_idx = max(max_idx, int(bullet_id.split("-", 1)[1]))
            except Exception:
                continue
        return f"{prefix}-{max_idx + 1:05d}"

    def get_stats_text(self) -> str:
        stats = self._ensure_statistics_defaults()
        lines = [
            f"total_bullets={stats['total_bullets']}",
            f"unique_strategies={stats['unique_strategies']}",
            f"total_executions={stats['total_executions']}",
            f"last_reflection={stats['last_reflection']}",
        ]
        section_counts = {section: 0 for section in SECTION_ORDER}
        for bullet in self.get_active_bullets():
            section_counts[bullet.section] += 1
        for section in SECTION_ORDER:
            lines.append(f"{section}={section_counts[section]}")
        return "\n".join(lines)

    def get(self, bullet_id: str) -> Bullet | None:
        return self.bullets.get(str(bullet_id))

    def remove(self, bullet_id: str) -> bool:
        if str(bullet_id) not in self.bullets:
            return False
        del self.bullets[str(bullet_id)]
        return self.save()

    def get_all(self) -> list[Bullet]:
        return list(self.bullets.values())

    def get_active_bullets(self) -> list[Bullet]:
        active: list[Bullet] = []
        for bullet in self.bullets.values():
            if bullet.status != "active":
                continue
            if bullet.harmful_count > 3 and bullet.harmful_count > bullet.helpful_count * 2:
                continue
            active.append(bullet)
        return active

    def get_bullets_by_ids(self, bullet_ids: list[str]) -> list[Bullet]:
        result = []
        for bullet_id in bullet_ids:
            bullet = self.bullets.get(str(bullet_id))
            if bullet is not None:
                result.append(bullet)
        return result

    def retrieve(self, query_text: str = "", max_bullets: int = 15) -> list[Bullet]:
        bullets = self.get_active_bullets()
        now = datetime.now()

        def _score(bullet: Bullet) -> float:
            overlap = lexical_overlap_score(query_text, bullet.content) if query_text else 0.0
            helpful = float(bullet.helpful_count) * 1.5
            harmful = float(bullet.harmful_count) * -2.0
            recency_bonus = 0.0
            try:
                age_days = max((now - datetime.fromisoformat(bullet.last_used)).days, 0)
                recency_bonus = max(0.0, 1.0 - min(age_days / 30.0, 1.0))
            except Exception:
                recency_bonus = 0.0
            section_bonus = 0.5 if bullet.section in {"modality_rules", "validation_checklist", "tool_usage"} else 0.0
            return overlap * 8.0 + helpful + harmful + recency_bonus + section_bonus

        if query_text:
            ranked = sorted(bullets, key=_score, reverse=True)
        else:
            ranked = sorted(bullets, key=lambda item: (item.helpful_count, item.last_used), reverse=True)

        selected = [item for item in ranked[:max_bullets] if normalize_bullet_content(item.content)]
        self.statistics["last_query"] = query_text or None
        return selected

    def format_playbook(self, query_text: str = "", max_bullets: int = 15) -> tuple[str, list[str], dict[str, Any]]:
        selected = self.retrieve(query_text=query_text, max_bullets=max_bullets)
        if not selected:
            return "ACE Playbook 当前为空。", [], {"sections": {}, "query_text": query_text}

        grouped: dict[str, list[Bullet]] = {section: [] for section in SECTION_ORDER}
        for bullet in selected:
            grouped.setdefault(bullet.section, []).append(bullet)

        lines = [
            "# ACE Playbook",
            "先阅读相关条目，再执行任务；只使用真正相关的规则，不要机械套用全部条目。",
        ]
        if query_text:
            lines.append(f"# 当前任务焦点\n{query_text}")

        section_stats: dict[str, int] = {}
        for section in SECTION_ORDER:
            items = grouped.get(section) or []
            if not items:
                continue
            section_stats[section] = len(items)
            lines.append("")
            lines.append(f"## {SECTION_TITLES[section]}")
            for bullet in items:
                lines.append(
                    f"- [{bullet.id}] helpful={bullet.helpful_count} harmful={bullet.harmful_count} :: {bullet.content}"
                )

        return "\n".join(lines).strip(), [item.id for item in selected], {
            "sections": section_stats,
            "query_text": query_text,
            "selected_count": len(selected),
        }

    def get_context_string(self, max_bullets: int = 15, query_text: str = "") -> tuple[str, list[str]]:
        context, used_ids, _ = self.format_playbook(query_text=query_text, max_bullets=max_bullets)
        return context, used_ids

    def add_bullet(
        self,
        content: str,
        section: str | None = None,
        bullet_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        normalized = normalize_bullet_content(content)
        if not normalized:
            return {
                "created": False,
                "reason": "empty_content",
                "message": "策略内容为空，已跳过。",
            }

        final_section = normalize_section(section, normalized, bullet_type)
        for bullet in self.get_active_bullets():
            similarity = semantic_similarity(bullet.content, normalized)
            if similarity >= 0.96:
                bullet.last_used = datetime.now().isoformat()
                bullet.metadata.setdefault("duplicate_hits", 0)
                bullet.metadata["duplicate_hits"] += 1
                self.save()
                return {
                    "created": False,
                    "reason": "duplicate",
                    "bullet_id": bullet.id,
                    "section": bullet.section,
                    "message": "已存在相同或高度相似策略，跳过新增。",
                }

        bullet_id = self._generate_bullet_id(final_section)
        bullet = Bullet(
            id=bullet_id,
            content=normalized,
            section=final_section,
            metadata=dict(metadata or {}),
        )
        self.bullets[bullet_id] = bullet
        self.save()
        return {
            "created": True,
            "bullet_id": bullet_id,
            "section": final_section,
            "message": f"已添加到 {final_section}。",
        }

    def apply_feedback(self, bullet_ids: list[str], tag: str) -> dict[str, Any]:
        normalized_tag = "harmful" if str(tag) == "harmful" else "helpful"
        now = datetime.now().isoformat()
        updated = []
        missing = []

        for bullet_id in bullet_ids:
            bullet = self.bullets.get(str(bullet_id))
            if bullet is None:
                missing.append(str(bullet_id))
                continue
            if normalized_tag == "helpful":
                bullet.helpful_count += 1
            else:
                bullet.harmful_count += 1
            bullet.last_used = now
            updated.append(bullet.id)

        self.save()
        return {
            "updated_ids": updated,
            "missing_ids": missing,
            "tag": normalized_tag,
        }

    def update_counts(self, bullet_id: str, is_helpful: bool) -> bool:
        result = self.apply_feedback([bullet_id], tag="helpful" if is_helpful else "harmful")
        return bool(result["updated_ids"])

    def get_statistics(self) -> dict[str, Any]:
        b_list = list(self.bullets.values())
        return {
            "total_bullets": len(b_list),
            "helpful_bullets": sum(1 for b in b_list if b.helpful_count > 0),
            "harmful_bullets": sum(1 for b in b_list if b.harmful_count > 0),
            "archived_bullets": sum(1 for b in b_list if b.status != "active"),
            "total_executions": self.statistics.get("total_executions", 0),
            "unique_strategies": self.statistics.get("unique_strategies", 0),
            "last_reflection": self.statistics.get("last_reflection"),
        }

    def grow_and_refine(
        self,
        record_execution: bool = True,
        similarity_threshold: float = 0.93,
    ) -> dict[str, Any]:
        merged_pairs: list[tuple[str, str]] = []
        removed_ids: list[str] = []

        active = sorted(self.get_active_bullets(), key=lambda item: item.id)
        survivors: set[str] = set()
        for idx, left in enumerate(active):
            if left.id in survivors:
                continue
            for right in active[idx + 1:]:
                if right.id in survivors:
                    continue
                if semantic_similarity(left.content, right.content) < similarity_threshold:
                    continue

                keep = left if (left.score, left.helpful_count) >= (right.score, right.helpful_count) else right
                drop = right if keep is left else left
                keep.helpful_count += drop.helpful_count
                keep.harmful_count += drop.harmful_count
                keep.metadata.setdefault("merged_from", [])
                keep.metadata["merged_from"].append(drop.id)
                survivors.add(drop.id)
                merged_pairs.append((keep.id, drop.id))

        for drop_id in survivors:
            self.bullets.pop(drop_id, None)
            removed_ids.append(drop_id)

        for bullet_id, bullet in list(self.bullets.items()):
            if bullet.harmful_count > 3 and bullet.harmful_count > bullet.helpful_count * 2:
                bullet.status = "archived"
                removed_ids.append(bullet_id)

        stats = self._ensure_statistics_defaults()
        stats["duplicate_merges"] = int(stats.get("duplicate_merges", 0)) + len(merged_pairs)
        stats["last_refine_at"] = datetime.now().isoformat()
        self.save(record_execution=record_execution)

        return {
            "operation": "grow_and_refine",
            "removed_count": len(set(removed_ids)),
            "removed_ids": sorted(set(removed_ids)),
            "merged_count": len(merged_pairs),
            "merged_pairs": merged_pairs,
            "recorded_execution": record_execution,
        }


Playbook = ACEPlaybookManager
