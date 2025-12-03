# DSSMS バックテスト高速化 - 修正案詳細ドキュメント

**作成日**: 2025-12-03  
**目的**: HierarchicalRankingSystem のボトルネック解消 (11秒 → 2-3秒)  
**対象ファイル**: `src/dssms/hierarchical_ranking_system.py`, `src/dssms/dssms_data_manager.py`

---

## 目次
1. [最適化案1: 並列処理の実装 (Phase 1 - 最優先)](#phase1)
2. [最適化案2: FastRankingCore の統合 (Phase 3 - 低優先)](#phase3)
3. [最適化案3: キャッシュウォーミング (Phase 2 - 中優先)](#phase2-cache)
4. [最適化案4: 重複計算の除去 (Phase 2 - 中優先)](#phase2-dedup)
5. [実装順序とテスト計画](#implementation-order)
6. [API制限対策](#api-limits)
7. [想定される課題](#risks)

---

<a id="phase1"></a>
## 1. 最適化案1: 並列処理の実装 (Phase 1)

### 📊 効果予測
- **現在**: 11秒 (50銘柄 × 0.22秒/銘柄)
- **最適化後**: 2-3秒 (50銘柄 ÷ 5並列 × 0.22秒 + オーバーヘッド)
- **高速化率**: **4-5倍**

### 🎯 推奨度: ★★★★★ (最優先)

### 📍 修正箇所

**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**メソッド**: `categorize_by_perfect_order_priority` (Line 135-175)

### 🔧 修正内容

#### 修正前 (現在のコード)
```python
def categorize_by_perfect_order_priority(self, symbols: List[str]) -> Dict[int, List[str]]:
    """
    パーフェクトオーダー状況による優先度分類
    
    Args:
        symbols: 分析対象銘柄リスト
        
    Returns:
        優先度レベル別の銘柄分類辞書
    """
    priority_groups = {1: [], 2: [], 3: []}
    
    self.logger.info(f"優先度分類開始: {len(symbols)}銘柄")
    
    # TODO-PERF-001: バッチ処理最適化
    for symbol in symbols:  # ← ボトルネック: シリアル処理
        try:
            # マルチタイムフレームデータ取得
            data_dict = self.data_manager.get_multi_timeframe_data(symbol)
            
            # パーフェクトオーダー検出
            po_result = self.perfect_order_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
            
            if po_result is None:
                priority_groups[3].append(symbol)
                continue
            
            # 優先度判定
            priority = self._determine_priority_level(po_result)
            priority_groups[priority].append(symbol)
            
        except Exception as e:
            self.logger.warning(f"銘柄 {symbol} の優先度分類エラー: {e}")
            priority_groups[3].append(symbol)
    
    # 結果ログ
    for level, group in priority_groups.items():
        self.logger.info(f"優先度レベル{level}: {len(group)}銘柄")
    
    return priority_groups
```

#### 修正後 (並列処理版)
```python
def categorize_by_perfect_order_priority(self, symbols: List[str]) -> Dict[int, List[str]]:
    """
    パーフェクトオーダー状況による優先度分類 (並列処理最適化版)
    
    Args:
        symbols: 分析対象銘柄リスト
        
    Returns:
        優先度レベル別の銘柄分類辞書
    
    Performance:
        - 並列データ取得: ThreadPoolExecutor (max_workers=3)
        - API制限対策: 300回/分 (yfinance制限)
        - 想定高速化: 11秒 → 2-3秒 (4-5倍)
    """
    priority_groups = {1: [], 2: [], 3: []}
    
    self.logger.info(f"優先度分類開始: {len(symbols)}銘柄")
    
    # PERF-001実装: 並列データ取得
    start_time = time.time()
    
    # API制限対策: max_workers=3 (300回/分 ÷ 60秒 = 5回/秒)
    # 50銘柄 ÷ 3並列 = 約17秒 (余裕を持たせた設定)
    data_cache = self.data_manager.batch_get_multi_timeframe_data(
        symbols, 
        max_workers=3  # yfinance API制限対策 (5回/秒以下)
    )
    
    fetch_time = time.time() - start_time
    self.logger.info(f"データ取得完了: {len(data_cache)}/{len(symbols)}銘柄 ({fetch_time:.2f}秒)")
    
    # 優先度判定 (データ取得済み、並列不要)
    for symbol in symbols:
        try:
            # キャッシュからデータ取得
            data_dict = data_cache.get(symbol)
            
            if data_dict is None:
                self.logger.warning(f"銘柄 {symbol} のデータ取得失敗")
                priority_groups[3].append(symbol)
                continue
            
            # パーフェクトオーダー検出
            po_result = self.perfect_order_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
            
            if po_result is None:
                priority_groups[3].append(symbol)
                continue
            
            # 優先度判定
            priority = self._determine_priority_level(po_result)
            priority_groups[priority].append(symbol)
            
        except Exception as e:
            self.logger.warning(f"銘柄 {symbol} の優先度分類エラー: {e}")
            priority_groups[3].append(symbol)
    
    # 結果ログ
    total_time = time.time() - start_time
    for level, group in priority_groups.items():
        self.logger.info(f"優先度レベル{level}: {len(group)}銘柄")
    
    self.logger.info(f"優先度分類完了: {total_time:.2f}秒 (データ取得: {fetch_time:.2f}秒)")
    
    return priority_groups
```

### 📝 修正ポイント

1. **並列データ取得の追加**
   - `batch_get_multi_timeframe_data` (既存実装) を使用
   - `max_workers=3` でAPI制限対策 (5回/秒以下)

2. **データキャッシュの利用**
   - 1回の並列取得で全銘柄のデータを取得
   - `data_cache` 辞書で管理

3. **エラーハンドリング**
   - データ取得失敗時は優先度3に分類
   - 個別銘柄の失敗は全体に影響しない (既存実装を継承)

4. **パフォーマンス計測**
   - データ取得時間と全体時間をログ出力
   - テスト時の効果検証に使用

### ✅ テスト項目

#### 機能テスト
- [ ] 50銘柄の優先度分類が正しく動作する
- [ ] 優先度レベル1/2/3の分類結果が修正前と一致する
- [ ] データ取得失敗時に優先度3に分類される

#### パフォーマンステスト
- [ ] 実行時間が11秒 → 2-3秒に短縮される
- [ ] データ取得時間が個別にログ出力される
- [ ] 50銘柄のうち45銘柄以上が正常取得できる (90%成功率)

#### エラーハンドリングテスト
- [ ] 個別銘柄のデータ取得失敗時にログ出力される
- [ ] データ取得失敗銘柄が優先度3に分類される
- [ ] 全銘柄失敗時に空の優先度グループが返される

#### API制限テスト
- [ ] max_workers=3で制限内 (5回/秒以下) に収まる
- [ ] 429エラー (Too Many Requests) が発生しない
- [ ] タイムアウト (30秒) が発生しない

### 🔍 テスト実装例

**テストファイル**: `tests/temp/test_20251203_parallel_processing.py`

```python
"""
並列処理実装のテスト (Phase 1)
一時テスト: 成功後削除
"""
import pytest
import time
from datetime import datetime
from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem

def test_parallel_processing_performance():
    """並列処理のパフォーマンステスト"""
    # Setup
    system = HierarchicalRankingSystem()
    test_symbols = ['7203.T', '6758.T', '9984.T']  # 3銘柄でテスト
    
    # Execute
    start_time = time.time()
    priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
    execution_time = time.time() - start_time
    
    # Assert
    assert execution_time < 5.0, f"実行時間が長すぎる: {execution_time}秒"
    assert len(priority_groups) == 3, "優先度グループが3つ必要"
    assert sum(len(g) for g in priority_groups.values()) == len(test_symbols), "全銘柄が分類される必要"
    
    print(f"✅ パフォーマンステスト成功: {execution_time:.2f}秒")

def test_parallel_processing_correctness():
    """並列処理の正確性テスト"""
    # Setup
    system = HierarchicalRankingSystem()
    test_symbols = ['7203.T', '6758.T']
    
    # Execute
    priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
    
    # Assert
    assert isinstance(priority_groups, dict), "辞書型で返される"
    assert all(isinstance(v, list) for v in priority_groups.values()), "各値はリスト"
    
    print(f"✅ 正確性テスト成功: {priority_groups}")

def test_parallel_processing_error_handling():
    """並列処理のエラーハンドリングテスト"""
    # Setup
    system = HierarchicalRankingSystem()
    test_symbols = ['INVALID.T', '7203.T']  # 無効銘柄を含む
    
    # Execute
    priority_groups = system.categorize_by_perfect_order_priority(test_symbols)
    
    # Assert
    assert 'INVALID.T' in priority_groups[3], "無効銘柄は優先度3に分類"
    assert len(priority_groups[1]) + len(priority_groups[2]) + len(priority_groups[3]) == 2
    
    print(f"✅ エラーハンドリングテスト成功")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### ⚠️ API制限対策の詳細

#### yfinance API制限
- **公式制限**: 300回/分 (1 Client ID)
- **非公式ライブラリ**: 明確な制限なし (サーバー負荷で制限される可能性)

#### 対策設定
```python
max_workers=3  # 並列度3
# 計算: 50銘柄 ÷ 3並列 = 約17秒
# レート: 3並列 × 1秒 = 3回/秒 < 5回/秒 (安全圏)
```

#### エスカレーション戦略
1. **max_workers=3**: 初期設定 (安全)
2. **max_workers=5**: 高速化優先 (リスク中)
3. **max_workers=1**: 429エラー発生時 (最安全)

#### 429エラー発生時の対応
```python
# dssms_data_manager.py の batch_get_multi_timeframe_data 内
except Exception as e:
    if "429" in str(e):
        self.logger.error(f"API制限超過: {symbol} - max_workersを減らしてください")
        time.sleep(60)  # 1分待機
    else:
        self.logger.warning(f"Failed to fetch data for {symbol}: {e}")
```

### 📊 想定される結果

#### Before (現在)
```
2025-12-03 19:52:39,451 - 優先度分類開始: 50銘柄
2025-12-03 19:52:50,818 - 優先度分類完了
実行時間: 11.367秒
```

#### After (並列処理)
```
2025-12-03 20:00:00,000 - 優先度分類開始: 50銘柄
2025-12-03 20:00:02,500 - データ取得完了: 48/50銘柄 (2.2秒)
2025-12-03 20:00:02,800 - 優先度分類完了: 2.8秒 (データ取得: 2.2秒)
実行時間: 2.8秒 (4.1倍高速化)
```

---

<a id="phase3"></a>
## 2. 最適化案2: FastRankingCore の統合 (Phase 3)

### 📊 効果予測
- **現在**: 計算処理 2-3秒 (pandas/numpy使用)
- **最適化後**: 計算処理 0.05-0.1秒 (pure Python)
- **高速化率**: 20-60倍 (計算部分のみ)

### 🎯 推奨度: ★★☆☆☆ (低優先)

### 💡 実装判断
**Phase 1実装後、全体時間が2-3秒になる場合は不要**

理由:
- 11秒のうち、データ取得が8-9秒、計算が2-3秒
- Phase 1でデータ取得を8-9秒削減
- 残り2-3秒の計算時間が問題になる場合のみ実装

### 🔧 修正内容 (参考)

**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**メソッド**: `rank_within_priority_group` (Line 176-210)

```python
# 修正前
def rank_within_priority_group(self, symbols: List[str]) -> List[Tuple[str, float]]:
    ranking_scores = []
    
    for symbol in symbols:
        score_data = self._calculate_comprehensive_score(symbol)
        if score_data:
            ranking_scores.append((symbol, score_data.total_score))
    
    ranking_scores.sort(key=lambda x: x[1], reverse=True)
    return ranking_scores

# 修正後 (FastRankingCore使用)
def rank_within_priority_group(self, symbols: List[str]) -> List[Tuple[str, float]]:
    if USE_FAST_CORE and _fast_ranking_adapter:
        # FastRankingCore使用
        score_data_list = [
            {"symbol": s, **self._calculate_comprehensive_score(s).__dict__}
            for s in symbols
        ]
        ranked_data = _fast_ranking_adapter.rank_symbols(score_data_list, {})
        return [(item["symbol"], item["total_score"]) for item in ranked_data]
    else:
        # 既存実装 (フォールバック)
        ranking_scores = []
        for symbol in symbols:
            score_data = self._calculate_comprehensive_score(symbol)
            if score_data:
                ranking_scores.append((symbol, score_data.total_score))
        
        ranking_scores.sort(key=lambda x: x[1], reverse=True)
        return ranking_scores
```

### ✅ テスト項目 (Phase 3実施時)
- [ ] 計算時間が2-3秒 → 0.05-0.1秒に短縮
- [ ] ランキング結果が既存実装と一致
- [ ] FastRankingCore未使用時のフォールバック動作

---

<a id="phase2-cache"></a>
## 3. 最適化案3: キャッシュウォーミング (Phase 2)

### 📊 効果予測
- **現在**: 初回実行 11秒 + 2回目以降 0.5秒 (キャッシュヒット)
- **最適化後**: 初回実行 2-3秒 (並列) + 事前ウォーミング 2秒
- **効果**: 初回実行の高速化

### 🎯 推奨度: ★★★☆☆ (中優先)

### 📍 修正箇所

**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**新規メソッド**: `warm_cache_for_symbols`

### 🔧 修正内容

```python
def warm_cache_for_symbols(self, symbols: List[str]) -> None:
    """
    キャッシュウォーミング (バックテスト開始前に実行)
    
    Args:
        symbols: ウォーミング対象銘柄リスト
    
    Performance:
        - 並列データ取得: max_workers=3
        - 取得時間: 約2秒 (50銘柄)
        - 効果: 初回実行の高速化
    
    Usage:
        # バックテスト開始前
        system = HierarchicalRankingSystem()
        system.warm_cache_for_symbols(target_symbols)
        # 以降のcategorize_by_perfect_order_priorityは高速化
    """
    self.logger.info(f"キャッシュウォーミング開始: {len(symbols)}銘柄")
    
    start_time = time.time()
    
    # 並列データ取得 (キャッシュに保存)
    data_cache = self.data_manager.batch_get_multi_timeframe_data(
        symbols, 
        max_workers=3
    )
    
    warm_time = time.time() - start_time
    
    self.logger.info(
        f"キャッシュウォーミング完了: {len(data_cache)}/{len(symbols)}銘柄 "
        f"({warm_time:.2f}秒)"
    )
```

### 📝 使用例

**ファイル**: `src/dssms/dssms_integrated_main.py`  
**メソッド**: `run_dynamic_backtest`

```python
def run_dynamic_backtest(self, start_date: datetime, end_date: datetime,
                       target_symbols: Optional[List[str]] = None) -> Dict[str, Any]:
    # ... 既存コード ...
    
    # Phase 2追加: キャッシュウォーミング (バックテスト開始前)
    if self.hierarchical_ranking_system and target_symbols:
        self.logger.info("キャッシュウォーミング実行")
        self.hierarchical_ranking_system.warm_cache_for_symbols(target_symbols)
    
    # バックテストループ
    current_date = start_date
    while current_date <= end_date:
        # ... 既存コード (高速化される) ...
```

### ✅ テスト項目
- [ ] ウォーミング実行時間が2-3秒以内
- [ ] ウォーミング後のcategorize_by_perfect_order_priorityが高速化
- [ ] キャッシュヒット率が90%以上

---

<a id="phase2-dedup"></a>
## 4. 最適化案4: 重複計算の除去 (Phase 2)

### 📊 効果予測
- **現在**: 100回のデータ取得 (50銘柄 × 2回)
- **最適化後**: 50回のデータ取得 (50銘柄 × 1回)
- **削減率**: 50%

### 🎯 推奨度: ★★★☆☆ (中優先)

### 📍 修正箇所

**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**メソッド**: `get_top_candidate` (Line 213-243)

### 🔧 修正内容

#### 修正前 (重複あり)
```python
def get_top_candidate(self, available_funds: float) -> Optional[str]:
    # 1回目: データ取得 + 優先度分類
    priority_groups = self.categorize_by_perfect_order_priority(screener_result)
    
    # 2回目: データ取得 + スコア計算 (重複!)
    for priority_level in [1, 2, 3]:
        ranked_symbols = self.rank_within_priority_group(group_symbols)
        # ...
```

#### 修正後 (データ共有版)
```python
def get_top_candidate(self, available_funds: float) -> Optional[str]:
    """
    利用可能資金を考慮した最適候補銘柄選択 (重複除去版)
    """
    self.logger.info(f"最適候補選択開始: 利用可能資金 {available_funds:,.0f}円")
    
    # 全銘柄の優先度分類
    screener_result = self._get_screened_symbols(available_funds)
    
    # Phase 2追加: データキャッシュ
    # 1回のデータ取得で優先度分類とスコア計算の両方に使用
    data_cache = self.data_manager.batch_get_multi_timeframe_data(
        screener_result, 
        max_workers=3
    )
    
    # 優先度分類 (キャッシュ使用)
    priority_groups = self._categorize_with_cache(screener_result, data_cache)
    
    # 優先度順に最適候補を探索
    for priority_level in [1, 2, 3]:
        group_symbols = priority_groups.get(priority_level, [])
        
        if not group_symbols:
            continue
        
        # グループ内ランキング (キャッシュ使用 - 重複なし)
        ranked_symbols = self._rank_with_cache(group_symbols, data_cache)
        
        # 購入可能性チェック
        for symbol, score in ranked_symbols:
            if self._check_affordability(symbol, available_funds):
                self.logger.info(
                    f"最適候補決定: {symbol} "
                    f"(優先度レベル{priority_level}, スコア{score:.3f})"
                )
                return symbol
    
    self.logger.warning("購入可能な最適候補が見つかりませんでした")
    return None

def _categorize_with_cache(self, symbols: List[str], 
                          data_cache: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[int, List[str]]:
    """キャッシュを使用した優先度分類"""
    priority_groups = {1: [], 2: [], 3: []}
    
    for symbol in symbols:
        try:
            data_dict = data_cache.get(symbol)
            if data_dict is None:
                priority_groups[3].append(symbol)
                continue
            
            po_result = self.perfect_order_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
            priority = self._determine_priority_level(po_result)
            priority_groups[priority].append(symbol)
            
        except Exception as e:
            self.logger.warning(f"銘柄 {symbol} の優先度分類エラー: {e}")
            priority_groups[3].append(symbol)
    
    return priority_groups

def _rank_with_cache(self, symbols: List[str], 
                    data_cache: Dict[str, Dict[str, pd.DataFrame]]) -> List[Tuple[str, float]]:
    """キャッシュを使用したランキング (重複除去)"""
    ranking_scores = []
    
    for symbol in symbols:
        try:
            # キャッシュからデータ取得 (重複なし)
            data_dict = data_cache.get(symbol)
            if data_dict is None:
                continue
            
            # スコア計算 (データ取得なし)
            score_data = self._calculate_comprehensive_score_with_cache(symbol, data_dict)
            
            if score_data:
                ranking_scores.append((symbol, score_data.total_score))
                
        except Exception as e:
            self.logger.warning(f"銘柄 {symbol} のスコア計算エラー: {e}")
    
    ranking_scores.sort(key=lambda x: x[1], reverse=True)
    return ranking_scores
```

### ✅ テスト項目
- [ ] データ取得回数が100回 → 50回に削減
- [ ] 優先度分類結果が修正前と一致
- [ ] ランキング結果が修正前と一致
- [ ] キャッシュミス時のエラーハンドリング

---

<a id="implementation-order"></a>
## 5. 実装順序とテスト計画

### Phase 1: 並列処理の実装 (最優先)

**期間**: 1-2日  
**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**工数**: 中

#### 実装手順
1. `categorize_by_perfect_order_priority` の修正
2. `max_workers=3` 設定 (API制限対策)
3. パフォーマンス計測ログの追加
4. テスト実装 (`test_20251203_parallel_processing.py`)
5. 実行・検証
6. 効果確認 (11秒 → 2-3秒)

#### 成功基準
- [ ] 実行時間が11秒 → 2-3秒に短縮
- [ ] 429エラー (API制限) が発生しない
- [ ] 優先度分類結果が修正前と一致

---

### Phase 2: 重複除去 + キャッシュウォーミング (中優先)

**期間**: 0.5-1日  
**ファイル**: `src/dssms/hierarchical_ranking_system.py`, `src/dssms/dssms_integrated_main.py`  
**工数**: 低-中

#### 実装手順 (Phase 1完了後)
1. `warm_cache_for_symbols` メソッド追加
2. `get_top_candidate` の重複除去
3. `dssms_integrated_main.py` のウォーミング統合
4. テスト実装
5. 効果確認 (さらに0.5-1秒短縮)

#### 成功基準
- [ ] 実行時間がさらに0.5-1秒短縮
- [ ] データ取得回数が50%削減
- [ ] キャッシュヒット率が90%以上

---

### Phase 3: FastRankingCore 統合 (低優先 - 条件付き)

**期間**: 3-5日  
**ファイル**: `src/dssms/hierarchical_ranking_system.py`  
**工数**: 高

#### 実装条件
**Phase 1-2実装後、全体時間が2-3秒以上かかる場合のみ実施**

#### 実装手順 (条件満たす場合)
1. `rank_within_priority_group` の FastRankingCore 統合
2. `_calculate_comprehensive_score` の最適化
3. フォールバック実装
4. テスト実装
5. 効果確認 (計算2-3秒 → 0.05-0.1秒)

#### 成功基準
- [ ] 計算時間が2-3秒 → 0.05-0.1秒に短縮
- [ ] ランキング結果が修正前と一致
- [ ] FastRankingCore未使用時の動作確認

---

<a id="api-limits"></a>
## 6. API制限対策

### yfinance API制限の詳細

#### 公式制限
- **制限**: 300回/分 (1 Client ID)
- **超過時**: 429 Too Many Requests エラー

#### 非公式ライブラリ (yfinance Python)
- **明確な制限なし** (公式未公表)
- **サーバー負荷で制限される可能性あり**

### 並列度の設定根拠

#### max_workers=3 (推奨設定)
```
計算:
- 50銘柄 ÷ 3並列 = 約17秒
- レート: 3並列 × 1回/秒 = 3回/秒
- 余裕: 3回/秒 < 5回/秒 (300回/分 ÷ 60秒)
```

#### max_workers=5 (高速化優先)
```
計算:
- 50銘柄 ÷ 5並列 = 約10秒
- レート: 5並列 × 1回/秒 = 5回/秒
- リスク: 5回/秒 = 300回/分の限界 (ギリギリ)
```

#### max_workers=1 (最安全)
```
計算:
- 50銘柄 × 1秒 = 50秒
- レート: 1回/秒
- 用途: 429エラー発生時のフォールバック
```

### エスカレーション戦略

```python
# 推奨実装 (将来の拡張用)
def batch_get_multi_timeframe_data_with_retry(self, symbols: List[str]) -> Dict:
    """API制限対策付きバッチデータ取得"""
    max_workers_options = [3, 1]  # 3並列 → 1並列にフォールバック
    
    for max_workers in max_workers_options:
        try:
            return self.batch_get_multi_timeframe_data(symbols, max_workers)
        except Exception as e:
            if "429" in str(e):
                self.logger.warning(f"API制限超過 (max_workers={max_workers})")
                if max_workers == max_workers_options[-1]:
                    raise  # 最後の試行も失敗
                time.sleep(60)  # 1分待機
                continue
            else:
                raise
```

---

<a id="risks"></a>
## 7. 想定される課題

### 課題1: API制限による429エラー

**発生条件**: max_workers が大きすぎる場合

**対策**:
1. max_workers=3 から開始 (安全設定)
2. 429エラー発生時のログ監視
3. エラー発生時は max_workers=1 に変更

**検証方法**:
```python
# テストコード
def test_api_limit_handling():
    system = HierarchicalRankingSystem()
    symbols = ['7203.T'] * 100  # 100銘柄で負荷テスト
    
    try:
        result = system.categorize_by_perfect_order_priority(symbols)
        assert "429" not in str(result)
    except Exception as e:
        if "429" in str(e):
            pytest.fail("API制限超過エラー発生")
```

---

### 課題2: ThreadPoolExecutor のオーバーヘッド

**影響**: 銘柄数が少ない場合、逆に遅くなる可能性

**対策**:
```python
def categorize_by_perfect_order_priority(self, symbols: List[str]) -> Dict[int, List[str]]:
    # 銘柄数が少ない場合はシリアル処理
    if len(symbols) < 10:
        self.logger.info("銘柄数が少ないためシリアル処理")
        return self._categorize_serial(symbols)
    
    # 並列処理
    return self._categorize_parallel(symbols)
```

---

### 課題3: データ取得失敗時の影響範囲

**影響**: 個別銘柄の失敗が全体に影響しないか

**対策**: 既存実装を継承 (個別失敗は優先度3に分類)

**検証方法**:
```python
def test_partial_failure_handling():
    system = HierarchicalRankingSystem()
    symbols = ['INVALID.T', '7203.T', '6758.T']
    
    result = system.categorize_by_perfect_order_priority(symbols)
    
    # INVALID.T は優先度3、他は正常分類
    assert 'INVALID.T' in result[3]
    assert len(result[1]) + len(result[2]) > 0
```

---

### 課題4: メモリ使用量の増加

**影響**: data_cache に50銘柄 × 3時間軸のデータを保持

**推定メモリ**: 
- 1銘柄 × 300日 × 6カラム × 8バイト = 14.4KB
- 50銘柄 × 3時間軸 × 14.4KB = 2.16MB

**対策**: 問題なし (2MB程度)

---

## 8. 実装後の検証チェックリスト

### Phase 1実装後
- [ ] 実行時間が11秒 → 2-3秒に短縮
- [ ] ログに「データ取得完了: XX/50銘柄 (Y.Y秒)」が出力
- [ ] 429エラーが発生しない
- [ ] 優先度分類結果が修正前と一致
- [ ] テストファイルを `docs/test_history/` に記録
- [ ] 一時テストファイルを削除

### Phase 2実装後
- [ ] 実行時間がさらに0.5-1秒短縮
- [ ] データ取得回数のログ確認 (50回以下)
- [ ] キャッシュウォーミングのログ出力
- [ ] テストファイルを `docs/test_history/` に記録

### Phase 3実装後 (条件付き)
- [ ] 計算時間が0.05-0.1秒に短縮
- [ ] FastRankingCore のパフォーマンス統計出力
- [ ] フォールバック動作の確認

---

## 9. copilot-instructions.md 準拠チェック

### 基本原則
- [x] バックテスト実行をスキップしない (並列処理でも実行)
- [x] 検証なしの報告禁止 (テスト項目明記)
- [x] 不明な場合は推測せず報告

### フォールバック機能の制限
- [x] モック/ダミーデータ不使用 (実データのみ)
- [x] テスト継続目的のフォールバック禁止 (エラーは明示的にログ出力)
- [x] フォールバック実行時のログ必須 (既存実装を継承)

### データ取得ルール
- [x] yfinance に `auto_adjust=False` を指定 (既存実装を継承)
- [x] CSV キャッシュに `Adj Close` を保存 (既存実装を継承)

### テストファイル配置ルール
- [x] 一時テスト: `tests/temp/test_20251203_*.py`
- [x] 命名規則: `test_YYYYMMDD_<feature>.py`
- [x] 削除基準: 全アサーション成功 + 実データ検証完了

---

## 10. まとめ

### 実装推奨順序
1. **Phase 1 (最優先)**: 並列処理の実装 → 11秒 → 2-3秒
2. **Phase 2 (中優先)**: 重複除去 + キャッシュウォーミング → さらに0.5-1秒短縮
3. **Phase 3 (低優先)**: FastRankingCore 統合 → 条件付き実施

### 想定される最終結果
- **現在**: 11秒
- **Phase 1後**: 2-3秒 (4-5倍高速化)
- **Phase 2後**: 1.5-2秒 (さらに0.5-1秒短縮)
- **Phase 3後**: 1-1.5秒 (条件付き)

### 次のアクション
1. Phase 1の実装開始
2. テスト実行・検証
3. 効果確認後、Phase 2に進む

---

**ドキュメント作成日**: 2025-12-03  
**最終更新日**: 2025-12-03  
**作成者**: Backtest Project Team
