"""
Module: Minimum Weight Rule Manager
File: minimum_weight_rule_manager.py
Description:
  3-2-2 階層的最小重み設定ルール管理クラス
  ルールのロード・保存・編集・バリデーション（Python型・値チェックのみ）

Author: imega
Created: 2025-07-17
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from config.portfolio_weight_calculator import MinimumWeightRule, MinimumWeightLevel

class MinimumWeightRuleManager:
    """階層的最小重みルール管理クラス"""
    def __init__(self, rules_file: Optional[str] = None):
        self.rules_file = Path(rules_file or "config/portfolio_weights/minimum_weights/minimum_weight_rules.json")
        self.strategy_rules: Dict[str, MinimumWeightRule] = {}
        self.category_rules: Dict[str, MinimumWeightRule] = {}
        self.default_rule: Optional[MinimumWeightRule] = None
        self._load_rules()

    def _load_rules(self):
        if not self.rules_file.exists():
            return
        try:
            with open(self.rules_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 型・値チェック
            self.strategy_rules = {}
            for rule_data in data.get("strategy_rules", []):
                rule = self._dict_to_rule(rule_data)
                self.strategy_rules[rule.strategy_name] = rule
            self.category_rules = {}
            for rule_data in data.get("category_rules", []):
                rule = self._dict_to_rule(rule_data)
                if rule.category:
                    self.category_rules[rule.category] = rule
            default_data = data.get("default_rule")
            if default_data:
                self.default_rule = self._dict_to_rule(default_data)
        except Exception as e:
            print(f"ルール読み込みエラー: {e}")

    def _save_rules(self):
        try:
            data = {
                "strategy_rules": [self._rule_to_dict(rule) for rule in self.strategy_rules.values()],
                "category_rules": [self._rule_to_dict(rule) for rule in self.category_rules.values()],
                "default_rule": self._rule_to_dict(self.default_rule) if self.default_rule else None
            }
            with open(self.rules_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"ルール保存エラー: {e}")

    def _dict_to_rule(self, data: Dict[str, Any]) -> MinimumWeightRule:
        try:
            min_weight = float(data["min_weight"])
            if not (0.0 <= min_weight <= 1.0):
                raise ValueError("min_weight範囲エラー")
            level = MinimumWeightLevel(data["level"])
            exclusion_threshold = data.get("exclusion_threshold")
            if exclusion_threshold is not None:
                exclusion_threshold = float(exclusion_threshold)
                if not (0.0 <= exclusion_threshold <= min_weight):
                    raise ValueError("exclusion_threshold範囲エラー")
            return MinimumWeightRule(
                strategy_name=str(data["strategy_name"]),
                min_weight=min_weight,
                level=level,
                category=data.get("category"),
                is_conditional=bool(data.get("is_conditional", False)),
                conditions=dict(data.get("conditions", {})),
                exclusion_threshold=exclusion_threshold,
                created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
            )
        except Exception as e:
            print(f"型・値チェックエラー: {e}")
            raise

    def _rule_to_dict(self, rule: MinimumWeightRule) -> Dict[str, Any]:
        return {
            "strategy_name": rule.strategy_name,
            "min_weight": rule.min_weight,
            "level": rule.level.value,
            "category": rule.category,
            "is_conditional": rule.is_conditional,
            "conditions": rule.conditions,
            "exclusion_threshold": rule.exclusion_threshold,
            "created_at": rule.created_at.isoformat()
        }

    def add_strategy_rule(self, rule: MinimumWeightRule) -> bool:
        self.strategy_rules[rule.strategy_name] = rule
        self._save_rules()
        return True

    def add_category_rule(self, category: str, min_weight: float) -> bool:
        rule = MinimumWeightRule(
            strategy_name=f"category_{category}",
            min_weight=min_weight,
            level=MinimumWeightLevel.CATEGORY,
            category=category
        )
        self.category_rules[category] = rule
        self._save_rules()
        return True

    def set_default_rule(self, min_weight: float) -> bool:
        self.default_rule = MinimumWeightRule(
            strategy_name="default",
            min_weight=min_weight,
            level=MinimumWeightLevel.PORTFOLIO_DEFAULT
        )
        self._save_rules()
        return True

    def delete_strategy_rule(self, strategy_name: str) -> bool:
        if strategy_name in self.strategy_rules:
            del self.strategy_rules[strategy_name]
            self._save_rules()
            return True
        return False

    def delete_category_rule(self, category: str) -> bool:
        if category in self.category_rules:
            del self.category_rules[category]
            self._save_rules()
            return True
        return False

    def get_minimum_weight(self, strategy_name: str, category: Optional[str] = None) -> float:
        if strategy_name in self.strategy_rules:
            return self.strategy_rules[strategy_name].min_weight
        if category and category in self.category_rules:
            return self.category_rules[category].min_weight
        if self.default_rule:
            return self.default_rule.min_weight
        return 0.05

    def get_exclusion_threshold(self, strategy_name: str, category: Optional[str] = None) -> Optional[float]:
        if strategy_name in self.strategy_rules:
            return self.strategy_rules[strategy_name].exclusion_threshold
        if category and category in self.category_rules:
            return self.category_rules[category].exclusion_threshold
        return None

    def get_rules_summary(self) -> Dict[str, Any]:
        return {
            "strategy_rules": {name: self._rule_to_dict(rule) for name, rule in self.strategy_rules.items()},
            "category_rules": {name: self._rule_to_dict(rule) for name, rule in self.category_rules.items()},
            "default_rule": self._rule_to_dict(self.default_rule) if self.default_rule else None
        }
