# DSSMS パーフェクトオーダー優先スコアリング統合計画

**作成日**: 2025-12-15  
**ステータス**: 実装準備完了  
**優先度**: 最高  

---

## 1. 目的・目標

### **問題の本質**
- 現状: 20銘柄中10銘柄（50%）がPO条件未達（レベル3）
- 原因: Stage A（最終選択）がPO条件を完全に無視
- 目標: レベル3を5銘柄以下（25%以下）に削減

### **解決アプローチ**
パーフェクトオーダー（PO）を**主要評価軸（60-70%）**とし、時価総額・モメンタム・出来高・ボラティリティを補助評価軸（30-40%）として統合

### **期待効果**
- **品質向上**: PO条件達成銘柄の優先選択
- **時間短縮**: 50→20銘柄（54-60%削減、9000ms→4140ms）
- **根本解決**: Stage AとStage Bのスコアリング一貫性確保

---

## 2. 判明したこと（調査結果）

### **2.1 Stage A: algorithm_optimization_integration.py**
**根拠**: L143-153

```python
# 現在のスコアリング構成
scores = (
    market_cap_scores * 0.40 +          # 時価総額: 40%
    price_momentum_scores * 0.30 +       # モメンタム: 30%
    volume_scores * 0.20 -               # 出来高: 20%
    volatility_penalties * 0.10          # ボラティリティ: -10%
)
# PO条件: 0%（含まれていない）
```

**問題**: POスコアが存在しない

---

### **2.2 Stage B: hierarchical_ranking_system.py**
**根拠**: L383-387, L111-115

```python
# scoring_weights設定
{
    "fundamental": 0.40,   # 40%
    "technical": 0.30,     # 30%
    "volume": 0.20,        # 20%
    "volatility": 0.10     # 10%
    # "perfect_order": なし（欠落）
}

# 総合スコア計算（L383-387）
total_score = (
    fundamental_score * 0.40 +
    technical_score * 0.30 +
    volume_score * 0.20 +
    volatility_score * 0.10
)
# perfect_order_scoreは計算されているが（L379）、含まれていない
```

**問題**: 
1. `perfect_order_score` を計算するが使用しない（設計バグ）
2. `scoring_weights` に `"perfect_order"` キーが存在しない

---

### **2.3 POデータ取得機能**
**根拠**: perfect_order_detector.py, dssms_data_manager.py

**確認事実**:
- `PerfectOrderDetector.check_multi_timeframe_perfect_order()` 実装済み
- `DSSMSDataManager.get_multi_timeframe_data()` 実装済み
- 日足・週足・月足の3軸でPO判定・スコア算出が可能

**結論**: POデータは既に利用可能（統合のみ必要）

---

### **2.4 データフロー確認**

**現在**:
```
Stage A (algorithm_optimization_integration.py)
  ↓ 時価総額40% + モメンタム30% + 出来高20% - ボラティリティ10%
  ↓ PO条件: 0%
  ↓
20銘柄選択（レベル3が50%混入）
  ↓
Stage B (hierarchical_ranking_system.py)
  ↓ POスコア計算（未使用）
  ↓
最終1銘柄選択
```

**提案後**:
```
Stage A (algorithm_optimization_integration.py)
  ↓ PO条件: 65% ★追加
  ↓ 時価総額15% + モメンタム10% + 出来高5% + ボラティリティ5%
  ↓
20銘柄選択（PO優先、レベル3が25%以下）
  ↓
Stage B (hierarchical_ranking_system.py)
  ↓ 総合スコア計算（POスコア含む） ★修正
  ↓
最終1銘柄選択
```

---

## 3. 設計・修正案

### **3.1 スコアリング重み配分（提案）**

#### **Stage A: 最終選択（algorithm_optimization_integration.py）**
```python
weights = {
    'perfect_order': 0.65,      # PO条件: 65% ★新規
    'market_cap': 0.15,         # 時価総額: 15% (40% → 15%)
    'price_momentum': 0.10,     # モメンタム: 10% (30% → 10%)
    'volume_score': 0.05,       # 出来高: 5% (20% → 5%)
    'volatility_penalty': 0.05  # ボラティリティ: 5% (10% → 5%)
}
# 合計: 100%
```

**根拠**: 
- PO条件を主要評価軸に（60-70%）
- 既存指標も維持（バランス保持）
- Stage Bとの一貫性確保

---

#### **Stage B: グループ内ランキング（hierarchical_ranking_system.py）**
```python
scoring_weights = {
    "perfect_order": 0.30,   # PO強度: 30% ★新規
    "fundamental": 0.30,     # ファンダ: 30% (40% → 30%)
    "technical": 0.20,       # テクニカル: 20% (30% → 20%)
    "volume": 0.10,          # 出来高: 10% (20% → 10%)
    "volatility": 0.10       # ボラティリティ: 10% (維持)
}
# 合計: 100%
```

**根拠**:
- POスコアを総合スコアに統合（バグ修正）
- Stage Aよりもバランス重視（詳細分析のため）

---

### **3.2 技術的実装方針**

#### **A. PO データ取得の統合**
- `_parallel_data_collection()` 内で `PerfectOrderDetector` を呼び出し
- `DSSMSDataManager.get_multi_timeframe_data()` でデータ取得
- `check_multi_timeframe_perfect_order()` でPOスコア算出

#### **B. スコア正規化**
- POスコアは既に0.0-1.0の範囲（strength_score）
- 既存スコアと同じ正規化手法を適用

#### **C. パフォーマンス最適化**
- 並列処理を維持（ThreadPoolExecutor）
- キャッシュ機能活用（days_back=400で取得済み）

---

## 4. 実装タスク

### **Task 1: バックアップ作成** ⏱️ 5分
**優先度**: 最高  
**前提条件**: なし

**実行内容**:
```powershell
# バックアップディレクトリ作成
New-Item -Path "backups/20251215_po_integration" -ItemType Directory -Force

# 4ファイルをバックアップ
Copy-Item "src/dssms/algorithm_optimization_integration.py" "backups/20251215_po_integration/"
Copy-Item "config/dssms/dssms_config.json" "backups/20251215_po_integration/"
Copy-Item "src/dssms/hierarchical_ranking_system.py" "backups/20251215_po_integration/"
Copy-Item "src/dssms/nikkei225_screener.py" "backups/20251215_po_integration/"
```

**成功基準**:
- [×] 4ファイル全てバックアップ済み
- [×] バックアップファイルが読み取り可能

---

### **Task 2: algorithm_optimization_integration.py 修正** ⏱️ 20分
**優先度**: 最高  
**前提条件**: Task 1完了

#### **2.1 PerfectOrderDetector インポート追加**
**修正箇所**: L1-13（インポート部分）

```python
# 追加インポート
from .perfect_order_detector import PerfectOrderDetector
from .dssms_data_manager import DSSMSDataManager
```

---

#### **2.2 __init__() メソッド修正**
**修正箇所**: L18-24

```python
def __init__(self, logger: Optional[logging.Logger] = None):
    self.logger = logger or logging.getLogger(__name__)
    self.optimization_stats = {
        'numpy_operations': 0,
        'vectorized_calculations': 0,
        'early_terminations': 0,
        'processing_time_saved': 0.0
    }
    
    # PO検出器の初期化 ★追加
    self.po_detector = PerfectOrderDetector()
    self.data_manager = DSSMSDataManager()
```

---

#### **2.3 optimized_final_selection() デフォルト重み変更**
**修正箇所**: L49-54

```python
# デフォルト重み設定
weights = scoring_weights or {
    'perfect_order': 0.65,      # ★追加
    'market_cap': 0.15,         # 0.4 → 0.15
    'price_momentum': 0.10,     # 0.3 → 0.10
    'volume_score': 0.05,       # 0.2 → 0.05
    'volatility_penalty': 0.05  # 0.1 → 0.05
}
```

---

#### **2.4 _parallel_data_collection() にPOデータ追加**
**修正箇所**: L93-109

```python
def fetch_symbol_data(symbol: str) -> Optional[Dict[str, Any]]:
    try:
        # キャッシュ統合データ取得
        market_cap = market_data_fetcher.get_market_cap_data_cached(symbol)
        price = market_data_fetcher.get_price_data_cached(symbol)
        volume = market_data_fetcher.get_volume_data_cached(symbol)
        
        if any(data is None for data in [market_cap, price, volume]):
            return None
        
        # ★POスコア取得追加
        try:
            data_dict = self.data_manager.get_multi_timeframe_data(symbol, days_back=400)
            po_result = self.po_detector.check_multi_timeframe_perfect_order(symbol, data_dict)
            
            # 3軸のPO強度を統合
            if po_result:
                daily_strength = po_result.daily_result.strength_score if po_result.daily_result else 0.0
                weekly_strength = po_result.weekly_result.strength_score if po_result.weekly_result else 0.0
                monthly_strength = po_result.monthly_result.strength_score if po_result.monthly_result else 0.0
                
                # 加重平均（長期重視）
                po_score = (daily_strength * 0.3 + weekly_strength * 0.4 + monthly_strength * 0.3)
            else:
                po_score = 0.0
        except Exception as e:
            self.logger.debug(f"PO score calculation failed for {symbol}: {e}")
            po_score = 0.0
        
        return {
            'symbol': symbol,
            'market_cap': market_cap,
            'price': price,
            'volume': volume,
            'price_momentum': self._calculate_momentum(symbol, market_data_fetcher),
            'volatility': self._calculate_volatility(symbol, market_data_fetcher),
            'perfect_order_score': po_score  # ★追加
        }
        
    except Exception as e:
        self.logger.debug(f"Data collection failed for {symbol}: {e}")
        return None
```

---

#### **2.5 _vectorized_scoring() にPOスコア統合**
**修正箇所**: L133-153

```python
def _vectorized_scoring(
    self, 
    symbol_data: List[Dict[str, Any]], 
    weights: Dict[str, float]
) -> np.ndarray:
    """NumPy vectorized scoring計算"""
    
    self.optimization_stats['numpy_operations'] += 1
    
    n_symbols = len(symbol_data)
    
    # データをNumPy配列に変換
    market_caps = np.array([data['market_cap'] for data in symbol_data])
    prices = np.array([data['price'] for data in symbol_data])
    volumes = np.array([data['volume'] for data in symbol_data])
    momentums = np.array([data.get('price_momentum', 0.0) for data in symbol_data])
    volatilities = np.array([data.get('volatility', 0.0) for data in symbol_data])
    po_scores = np.array([data.get('perfect_order_score', 0.0) for data in symbol_data])  # ★追加
    
    # 正規化（0-1スケール）
    market_cap_scores = self._normalize_array(market_caps)
    price_momentum_scores = self._normalize_array(momentums)
    volume_scores = self._normalize_array(volumes)
    volatility_penalties = self._normalize_array(volatilities, reverse=True)
    # POスコアは既に0-1範囲なので正規化不要 ★追加
    
    # 重み付きスコア計算
    scores = (
        po_scores * weights.get('perfect_order', 0.0) +  # ★追加
        market_cap_scores * weights.get('market_cap', 0.0) +
        price_momentum_scores * weights.get('price_momentum', 0.0) +
        volume_scores * weights.get('volume_score', 0.0) -
        volatility_penalties * weights.get('volatility_penalty', 0.0)
    )
    
    self.optimization_stats['vectorized_calculations'] += n_symbols
    
    return scores
```

**成功基準**:
- [ ] 全修正箇所の実装完了
- [ ] インポートエラーなし
- [ ] 構文エラーなし

---

### **Task 3: config/dssms/dssms_config.json 修正** ⏱️ 5分
**優先度**: 高  
**前提条件**: Task 1完了

**修正箇所**: JSONファイル全体

```json
{
  "screening": {
    "nikkei225_filters": {
      "min_price": 500,
      "min_market_cap": 10000000000,
      "min_trading_volume": 100000,
      "max_symbols": 20,
      "min_shares_affordable": 100
    },
    "perfect_order": {
      "daily": {"short": 5, "medium": 25, "long": 75},
      "weekly": {"short": 13, "medium": 26, "long": 52},
      "monthly": {"short": 9, "medium": 24, "long": 60}
    },
    "priority_logic": {
      "level_1": "all_timeframes_perfect_order",
      "level_2": "monthly_weekly_perfect_order",
      "level_3": "others"
    }
  },
  "ranking_system": {
    "scoring_weights": {
      "perfect_order": 0.30,
      "fundamental": 0.30,
      "technical": 0.20,
      "volume": 0.10,
      "volatility": 0.10
    }
  },
  "algorithm_optimization": {
    "scoring_weights": {
      "perfect_order": 0.65,
      "market_cap": 0.15,
      "price_momentum": 0.10,
      "volume_score": 0.05,
      "volatility_penalty": 0.05
    }
  },
  "fundamental_analysis": {
    "required_quarters": 3,
    "min_operating_margin": 0.02,
    "stability_threshold": 0.8,
    "growth_threshold": 0.05
  },
  "data_sources": {
    "nikkei225_list": "nikkei225_components.json",
    "backup_method": "yahoo_finance_api"
  },
  "cache_settings": {
    "timeout_minutes": 5,
    "max_entries": 1000
  }
}
```

**変更点**:
- `"ranking_system"` セクション追加（Stage B用）
- `"algorithm_optimization"` セクション追加（Stage A用）

**成功基準**:
- [ ] JSON構文エラーなし
- [ ] 全キーが正しく追加されている

---

### **Task 4: hierarchical_ranking_system.py 修正** ⏱️ 15分
**優先度**: 高  
**前提条件**: Task 1完了

#### **4.1 scoring_weights 設定変更**
**修正箇所**: L111-115

```python
# スコア重み設定
self.scoring_weights = config.get('ranking_system', {}).get('scoring_weights', {
    "perfect_order": 0.30,  # ★追加
    "fundamental": 0.30,    # 0.40 → 0.30
    "technical": 0.20,      # 0.30 → 0.20
    "volume": 0.10,         # 0.20 → 0.10
    "volatility": 0.10      # 維持
})
```

---

#### **4.2 _calculate_comprehensive_score() 修正（バグ修正）**
**修正箇所**: L383-387

```python
# 加重平均による総合スコア
total_score = (
    perfect_order_score * self.scoring_weights.get('perfect_order', 0.0) +  # ★追加
    fundamental_score * self.scoring_weights.get('fundamental', 0.0) +
    technical_score * self.scoring_weights.get('technical', 0.0) +
    volume_score * self.scoring_weights.get('volume', 0.0) +
    volatility_score * self.scoring_weights.get('volatility', 0.0)
)
```

**成功基準**:
- [ ] perfect_order_score が総合スコアに含まれる
- [ ] 重み合計が100%

---

### **Task 5: nikkei225_screener.py 修正** ⏱️ 5分
**優先度**: 中  
**前提条件**: Task 1完了

**修正箇所**: L421-425（呼び出し部分）

```python
# 6. Stage 3-2: OptimizedAlgorithmEngine最終選択
max_symbols = self.config["screening"]["nikkei225_filters"]["max_symbols"]
self.logger.info(f"[DEBUG] max_symbols設定値: {max_symbols}, 候補銘柄数: {len(symbols)}")
if len(symbols) > max_symbols:
    # 最適化された最終選択アルゴリズム使用
    # ★scoring_weightsパラメータ追加（オプション）
    scoring_weights = self.config.get("algorithm_optimization", {}).get("scoring_weights")
    
    symbols = self.algorithm_optimizer.optimized_final_selection(
        symbols=symbols,
        max_symbols=max_symbols,
        market_data_fetcher=self.cached_fetcher,
        scoring_weights=scoring_weights  # ★追加
    )
```

**成功基準**:
- [ ] scoring_weights が正しく渡される
- [ ] 既存動作が維持される

---

### **Task 6: 検証テスト実行** ⏱️ 15-20分
**優先度**: 最高  
**前提条件**: Task 2, 3, 4, 5完了

#### **6.1 内訳確認テスト**
```powershell
python tests\temp\test_20251215_max_symbols_20_breakdown.py
```

**成功基準**:
- [ ] エラーなく実行完了
- [ ] レベル1（全軸PO）: 10銘柄以上（50%以上）
- [ ] レベル2（月週PO）: 5銘柄前後（25%前後）
- [ ] レベル3（その他）: 5銘柄以下（25%以下）★目標達成
- [ ] 質スコア: 0.8以上

---

#### **6.2 結果分析**
**判定基準**:

| レベル3銘柄数 | 評価 | 判定 | 次のアクション |
|-------------|------|------|--------------|
| 0-2銘柄 | [EXCELLENT] | 目標超過達成 | Phase 4へ進む |
| 3-5銘柄 | [OK] | 目標達成 | Phase 4へ進む |
| 6-8銘柄 | [WARNING] | 改善必要 | 重み再調整検討 |
| 9銘柄以上 | [ERROR] | 目標未達 | 設計見直し |

---

### **Task 7: バックテスト動作確認** ⏱️ 10分
**優先度**: 高  
**前提条件**: Task 6完了、レベル3 ≤ 5銘柄達成

#### **7.1 簡易動作確認**
```python
# 簡易確認スクリプト作成
from src.dssms.hierarchical_ranking_system import HierarchicalRankingSystem

config = {}  # デフォルト設定使用
system = HierarchicalRankingSystem(config)

# スコアリング重み確認
print("Stage B Scoring Weights:")
print(system.scoring_weights)
```

**成功基準**:
- [ ] インポートエラーなし
- [ ] scoring_weights に "perfect_order" が含まれる
- [ ] 重み合計 = 1.0

---

#### **7.2 統合テスト（オプション）** ✓ 完了
**注意**: 時間がかかる場合はスキップ可

```powershell
# 短期バックテスト（1ヶ月）
# 旧コマンド（引数処理なし）: python src/dssms/dssms_backtester.py --start 2025-11-01 --end 2025-12-01
# 新コマンド（引数処理あり）: 
python -m src.dssms.dssms_integrated_main --start-date 2023-01-15 --end-date 2023-01-31
```

**成功基準**:
- [x] 実行エラーなし → Exit Code: 0、実行時間3.6秒
- [x] 取引件数 > 0 → 10件（BUY=10, SELL=10）、copilot-instructions.md準拠
- [x] パフォーマンス指標算出成功 → 総リターン+30.70%、勝率90%

**実行結果（2025-12-15実施）**:
- 期間: 2023-01-16 to 2023-01-31（12取引日）
- 取引件数: 10件（BUY=10, SELL=10）
- 総リターン: +30.70%（初期100万円 → 最終130.7万円）
- 勝率: 90.00%（9勝1敗）
- 銘柄切替: 2回（6506安川電機 → 6954ファナック）
- 出力: 10種類のファイル（CSV+JSON+TXT、Excel出力なし）
- 判定: **SUCCESS** ✓

---

## 5. リスク管理

### **5.1 ロールバック手順**
問題発生時は即座にバックアップから復元:

```powershell
# ロールバック実行
Copy-Item "backups/20251215_po_integration/*" "src/dssms/" -Force
Copy-Item "backups/20251215_po_integration/dssms_config.json" "config/dssms/" -Force
```

---

### **5.2 既知の制約**
1. **days_back=400必須**: 月足データ確保のため（12レコード以上必要）
2. **データキャッシュ**: 初回実行時にキャッシュ生成で時間がかかる
3. **API制限**: yfinance 300回/分を考慮

---

### **5.3 トラブルシューティング**

| 症状 | 原因 | 対処法 |
|-----|------|--------|
| ImportError: PerfectOrderDetector | インポートパス誤り | 相対インポート確認 |
| KeyError: 'perfect_order' | 設定ファイル未更新 | dssms_config.json確認 |
| レベル3が依然50% | POスコア未反映 | _vectorized_scoring確認 |
| 実行時間が2倍以上 | POデータ取得の並列化不足 | ThreadPoolExecutor設定確認 |

---

## 6. 次のステップ

### **Phase 5: パフォーマンス最適化（将来）**
目標達成後の追加改善:
1. POスコアのキャッシュ機能
2. 並列処理のチューニング
3. メモリ使用量の最適化

### **Phase 6: 長期バックテスト（将来）**
1年間のバックテストで効果検証

---

## 7. 参考情報

### **関連ドキュメント**
- `docs/dssms/`: DSSMS関連ドキュメント
- `.github/copilot-instructions.md`: コーディング規約

### **関連Issue/PR**
- 作成予定: "PO優先スコアリング統合 #001"

---

## 変更履歴
- 2025-12-15: 初版作成（調査完了）

---

**完了 - Task 1から7.2まで順次実行した**
