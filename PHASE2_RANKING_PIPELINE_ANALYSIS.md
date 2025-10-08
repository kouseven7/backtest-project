# Phase 2: ランキングパイプライン診断修正分析ドキュメント

## [TARGET] **概要**
`deterministic_removal_test_report.md`で発見された新問題「ランキングパイプライン診断の不安定性」の詳細分析と修正実装計画

## [CHART] **問題詳細分析**

### **症状: 構造不一致問題**
```log
初日    : keys=['date', 'rankings', 'top_symbol', 'top_score', 'total_symbols', 'data_source', 'diagnostic_info']
2日目以降: keys=['symbols', 'date']
```

### **根本原因分析**
1. **ランキング診断パイプラインの初回成功・後続失敗パターン**
2. **構造生成の不整合性**: 初日と2日目以降で異なる処理フロー
3. **診断成功率**: 1/10日 (10%) - 極めて低い安定性

### **影響範囲**
- `top_symbol=None` → 切替判定機能停止
- ISM信頼度低下: 0.8 → 0.4
- 切替回数激減: 目標30回 → 実際1回

## [SEARCH] **技術的根本原因**

### **疑われる問題箇所**
```python
# src/dssms/dssms_backtester.py の _update_symbol_ranking メソッド
def _update_symbol_ranking(self, current_date: datetime, symbols: List[str]) -> Optional[Dict]:
    """Resolution 19: ランキング診断・修復システム"""
    
    # [ALERT] 問題箇所1: 診断結果の構造不整合
    diagnostic_result = self._diagnose_ranking_pipeline(current_date, symbols)
    
    if not diagnostic_result.get('success', False):
        # [ALERT] 問題箇所2: 修復処理の不完全性
        return self._repair_failed_diagnostics(current_date, symbols)
    
    return diagnostic_result
```

### **構造不一致の発生メカニズム**
1. **初日**: `_diagnose_ranking_pipeline` が完全構造を生成
2. **2日目以降**: 診断失敗 → `_repair_failed_diagnostics` が簡易構造のみ生成
3. **結果**: 構造不一致による `top_symbol` 欠如

## 🛠️ **Phase 2 修正実装計画**

### **Priority 1: 構造統一システム実装**

#### **Target 1.1: ランキング結果構造統一**
```python
def _ensure_ranking_structure_consistency(self, ranking_result: Dict) -> Dict:
    """
    全日程で一貫したランキング構造を保証
    TODO(tag:phase2, rationale:構造統一): 初日/2日目以降の構造差異解消
    """
    required_keys = [
        'date', 'rankings', 'top_symbol', 'top_score', 
        'total_symbols', 'data_source', 'diagnostic_info'
    ]
    
    # 構造検証
    if not all(key in ranking_result for key in required_keys):
        logger.warning(f"[TOOL] 構造不整合検出: 欠如キー={set(required_keys) - set(ranking_result.keys())}")
        return self._repair_ranking_structure(ranking_result)
    
    return ranking_result

def _repair_ranking_structure(self, partial_result: Dict) -> Dict:
    """
    不完全なランキング結果を完全構造に修復
    """
    base_structure = {
        'date': partial_result.get('date'),
        'rankings': {},
        'top_symbol': None,
        'top_score': 0.0,
        'total_symbols': 0,
        'data_source': 'repaired_structure',
        'diagnostic_info': {'repair_applied': True}
    }
    
    # 既存データの統合
    if 'symbols' in partial_result:
        # symbols データから rankings 構造を再構築
        symbols = partial_result['symbols']
        base_structure['total_symbols'] = len(symbols)
        
        # ComprehensiveScoringEngine による再計算
        if symbols:
            scores = self._calculate_scores_for_symbols(symbols)
            base_structure['rankings'] = scores
            
            # top_symbol の決定
            if scores:
                top_symbol = max(scores.items(), key=lambda x: x[1])
                base_structure['top_symbol'] = top_symbol[0]
                base_structure['top_score'] = top_symbol[1]
    
    logger.info(f"[TOOL] 構造修復完了: top_symbol={base_structure['top_symbol']}")
    return base_structure
```

#### **Target 1.2: 診断安定性向上**
```python
def _stabilize_ranking_diagnostics(self, current_date: datetime, symbols: List[str]) -> Dict:
    """
    診断の安定性向上: 10% → 90% 成功率改善
    TODO(tag:phase2, rationale:診断安定化): 多段階フォールバック実装
    """
    diagnostic_attempts = [
        ('primary', self._primary_ranking_diagnosis),
        ('secondary', self._secondary_ranking_diagnosis),
        ('fallback', self._fallback_ranking_diagnosis)
    ]
    
    for attempt_name, diagnostic_func in diagnostic_attempts:
        try:
            result = diagnostic_func(current_date, symbols)
            if self._validate_ranking_structure(result):
                logger.info(f"[SEARCH] 診断成功: {attempt_name} - 構造完全性確認")
                return result
            else:
                logger.warning(f"[SEARCH] 診断部分成功: {attempt_name} - 構造修復が必要")
                return self._ensure_ranking_structure_consistency(result)
                
        except Exception as e:
            logger.warning(f"[SEARCH] 診断失敗: {attempt_name} - {str(e)}")
            continue
    
    # 全診断失敗時の緊急フォールバック
    logger.error("[ALERT] 全診断失敗 - 緊急フォールバック実行")
    return self._emergency_ranking_fallback(current_date, symbols)

def _emergency_ranking_fallback(self, current_date: datetime, symbols: List[str]) -> Dict:
    """
    全診断失敗時の緊急フォールバック
    ComprehensiveScoringEngine 直接利用による最低限ランキング生成
    """
    emergency_result = {
        'date': current_date,
        'rankings': {},
        'top_symbol': None,
        'top_score': 0.0,
        'total_symbols': len(symbols),
        'data_source': 'emergency_fallback',
        'diagnostic_info': {
            'emergency_mode': True,
            'all_diagnostics_failed': True,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    try:
        # ComprehensiveScoringEngine による直接スコア計算
        scores = {}
        for symbol in symbols:
            score = self.scoring_engine.calculate_composite_score(symbol)
            scores[symbol] = score
        
        emergency_result['rankings'] = scores
        
        if scores:
            top_symbol = max(scores.items(), key=lambda x: x[1])
            emergency_result['top_symbol'] = top_symbol[0]
            emergency_result['top_score'] = top_symbol[1]
            
        logger.info(f"[ALERT] 緊急フォールバック成功: top_symbol={emergency_result['top_symbol']}")
        
    except Exception as e:
        logger.error(f"[ALERT] 緊急フォールバック失敗: {str(e)}")
        # 最後の手段: ランダム選択
        if symbols:
            import random
            emergency_result['top_symbol'] = random.choice(symbols)
            emergency_result['top_score'] = 0.5
            emergency_result['diagnostic_info']['random_selection'] = True
    
    return emergency_result
```

### **Priority 2: エラー耐性強化**

#### **Target 2.1: 多段階診断システム**
```python
def _primary_ranking_diagnosis(self, current_date: datetime, symbols: List[str]) -> Dict:
    """主要診断: 既存の診断ロジック強化版"""
    # 既存の _diagnose_ranking_pipeline の改良版
    pass

def _secondary_ranking_diagnosis(self, current_date: datetime, symbols: List[str]) -> Dict:
    """副次診断: 代替手法による診断"""
    # 異なるアプローチでの診断実装
    pass

def _fallback_ranking_diagnosis(self, current_date: datetime, symbols: List[str]) -> Dict:
    """フォールバック診断: 最低限の機能保証"""
    # 確実に動作する簡易診断ロジック
    pass
```

#### **Target 2.2: 構造検証システム**
```python
def _validate_ranking_structure(self, result: Dict) -> bool:
    """ランキング結果構造の完全性検証"""
    required_keys = ['date', 'rankings', 'top_symbol', 'top_score', 'total_symbols']
    
    # 必須キー存在確認
    if not all(key in result for key in required_keys):
        return False
    
    # データ型検証
    validations = [
        isinstance(result['rankings'], dict),
        result['top_symbol'] is None or isinstance(result['top_symbol'], str),
        isinstance(result['top_score'], (int, float)),
        isinstance(result['total_symbols'], int),
        result['total_symbols'] >= 0
    ]
    
    return all(validations)
```

## [UP] **期待効果**

### **診断成功率改善**
- **現在**: 10% (1/10日)
- **目標**: 90%以上 (9/10日以上)

### **構造一致性**
- **現在**: 初日のみ完全構造
- **目標**: 全日程で統一構造

### **切替回数回復**
- **現在**: 1回/10日間
- **目標**: 3-5回/10日間 (30-50回/100日間相当)

### **システム安定性**
- **現在**: top_symbol=None による機能停止
- **目標**: 継続的な切替判定機能維持

## [TEST] **テスト戦略**

### **Phase 2 テストケース**
```python
def test_ranking_structure_consistency():
    """構造一致性テスト"""
    # 10日間のランキング結果構造が全て同一であることを検証
    pass

def test_diagnostic_success_rate():
    """診断成功率テスト"""
    # 100回実行中90回以上の診断成功を検証
    pass

def test_fallback_functionality():
    """フォールバック機能テスト"""
    # 全診断失敗時でも有効なランキング結果生成を検証
    pass
```

## [ROCKET] **実装順序**

### **Step 1**: 構造統一システム実装
- `_ensure_ranking_structure_consistency`
- `_repair_ranking_structure`

### **Step 2**: 診断安定性システム実装  
- `_stabilize_ranking_diagnostics`
- 多段階診断メソッド

### **Step 3**: エラー耐性強化
- `_emergency_ranking_fallback`
- 構造検証システム

### **Step 4**: 統合テスト・効果測定
- 10日間テスト実行
- 切替回数回復確認

## 📝 **実装メモ**

### **既存コード影響範囲**
- `src/dssms/dssms_backtester.py`: `_update_symbol_ranking` メソッド
- 診断パイプライン関連メソッド全般
- ComprehensiveScoringEngine 統合部分

### **注意事項**
- 既存の決定論的計算除去機能は維持
- ComprehensiveScoringEngine の実データ分析は保持
- ISM統合切替システムとの互換性確保

## [TARGET] **成功基準**

1. **診断成功率**: 90%以上達成
2. **構造一致性**: 全日程で統一構造確保
3. **切替回数**: 3-5回/10日間達成
4. **システム安定性**: top_symbol=None 問題完全解消

このPhase 2実装により、Phase 1で達成した決定論的計算除去効果を最大化し、DSSMS切替システムの本格的回復を実現する。