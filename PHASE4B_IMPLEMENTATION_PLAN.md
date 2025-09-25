# Phase 4B Implementation Plan: ISM Unified Switching Logic Overhaul

## 🎯 実装目標

**Phase 4A問題**: 構造修復成功も切替頻度激減 (1回のみ)
**Phase 4B目標**: ISM統合判定ロジック抜本的改善で切替頻度3-5倍向上

## 📊 Phase 4A問題分析結果

### 特定された根本問題
1. **ISM信頼度固定**: 0.4で固定され動的変化なし
2. **Market Volatility計算不全**: 常に0.0で市場変動検知不能
3. **統一切替判定硬直性**: daily/weekly共に過度に保守的
4. **ランキング結果無視**: 7203が1位でも6758固定選択

## 🔧 Phase 4B修正戦略

### Phase 4B-1: ISM信頼度計算動的化 (CRITICAL)

#### 現在の問題
```python
confidence: 0.4  # 固定値で判定能力なし
```

#### 修正内容
```python
# 動的信頼度計算
confidence = base_confidence * market_factor * time_factor * ranking_consistency
```

#### 実装場所
- `src/dssms/intelligent_switch_manager.py`
- `evaluate_all_switches()` メソッド内

### Phase 4B-2: Market Context計算強化 (HIGH)

#### 現在の問題
```python
'volatility': 0.0,  # 固定
'market_condition': 'normal'  # 固定
```

#### 修正内容
```python
# 実データ反映Market Context
volatility = calculate_real_volatility(historical_data)
market_condition = determine_dynamic_condition(trend, volatility, volume)
```

#### 実装場所
- ISM `_calculate_market_context()` メソッド新規追加
- 日次データ取得時に実行

### Phase 4B-3: 統一切替判定基準緩和 (HIGH)

#### 現在の問題
- Daily判定: 過度に保守的
- Weekly判定: 5日経過でも切替なし

#### 修正内容
```python
# 緩和された判定基準
if time_since_last_switch >= 3:  # 3日 → 強制検討
    switch_probability *= 1.5
if ranking_changed and confidence > 0.5:  # ランキング変化考慮
    return True
```

### Phase 4B-4: ランキング結果強制適用 (MEDIUM)

#### 現在の問題
```python
RANKINGS DICT FIRST: 7203  # ランキング1位
RANKING TOP_SYMBOL: 6758   # 実際選択 (不一致)
```

#### 修正内容
```python
# 強制ランキング適用
if ranking_result.get('rankings'):
    top_symbol = list(ranking_result['rankings'].keys())[0]
    # キャッシュよりリアルタイム結果優先
```

## 🚀 実装手順

### Step 1: ISM統合判定ロジック修正
1. `intelligent_switch_manager.py` 特定
2. `evaluate_all_switches()` メソッド修正
3. 動的信頼度計算アルゴリズム実装

### Step 2: Market Context計算エンジン追加
1. `_calculate_market_context()` 新規メソッド
2. Volatility計算ロジック追加
3. Market Condition判定拡張

### Step 3: 統一切替判定基準更新
1. Daily判定閾値緩和
2. Weekly判定条件追加
3. 時間経過切替強制実装

### Step 4: ランキング実行一致性システム
1. リアルタイム結果優先ロジック
2. キャッシュ依存削減
3. Top symbol決定プロセス修正

## 📈 期待効果

### 定量的目標
- **切替回数**: 1回 → 5-7回 (500%向上)
- **ISM信頼度**: 0.4固定 → 0.5-0.8動的変動
- **銘柄多様性**: 1銘柄固定 → 3-4銘柄切替

### 定性的改善
- **適応性向上**: 市場条件変化への柔軟対応
- **判定精度向上**: リアルタイムデータ反映
- **システム整合性**: ランキング結果と実行の一致

## 🧪 Phase 4Bテスト計画

### テストケース
1. **基本動作確認**: 10日間シミュレーション
2. **信頼度変動測定**: 日次信頼度ログ記録
3. **切替頻度検証**: 目標5-7回達成確認
4. **銘柄多様性確認**: 異なる銘柄への切替実行

### 成功基準
- ✅ 切替回数 ≥ 5回
- ✅ ISM信頼度 0.5-0.8範囲で変動
- ✅ Market volatility > 0.0記録
- ✅ ランキング1位銘柄選択実行

## 🔍 リスク評価

### 高リスク要素
1. **ISMロジック複雑化**: 新たなバグ発生可能性
2. **過度な切替**: 頻繁すぎる売買リスク
3. **互換性問題**: 既存システムとの統合課題

### 軽減策
1. **段階的実装**: Phase 4B-1から順次適用
2. **厳格なテスト**: 各段階で動作確認
3. **バックアップ保持**: 修正前コード保存

## 📋 実装チェックリスト

### Phase 4B-1 (CRITICAL)
- [ ] ISM信頼度計算ロジック修正
- [ ] 動的変動アルゴリズム実装
- [ ] テスト実行・検証

### Phase 4B-2 (HIGH)
- [ ] Market Context計算エンジン追加
- [ ] Volatility計算実装
- [ ] Market Condition判定拡張

### Phase 4B-3 (HIGH)
- [ ] 統一切替判定基準緩和
- [ ] Daily/Weekly判定更新
- [ ] 時間経過切替強制実装

### Phase 4B-4 (MEDIUM)
- [ ] ランキング結果強制適用
- [ ] リアルタイム結果優先
- [ ] Top symbol決定修正

## 💡 技術的考慮事項

### パフォーマンス最適化
- **計算効率**: 新規計算処理の最適化
- **メモリ使用**: 追加データ構造の効率化
- **実行時間**: 処理速度維持

### 保守性確保
- **コード可読性**: 明確なコメント追加
- **モジュール分離**: 機能別の適切な分割
- **テスト網羅性**: 全修正箇所のテスト実装

---

**Phase 4B実装により、DSSMS切替システムの根本的改善を目指し、Phase 3で特定された1回切替問題の完全解決を図る。**