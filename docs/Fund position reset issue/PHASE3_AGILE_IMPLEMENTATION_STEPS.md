# Phase 3 アジャイル実装ステップ: backtest_daily()リアルトレード対応システム

**作成日**: 2025年12月30日  
**目的**: Phase 3実装をアジャイル方式で安全・効率的に進めるための具体的実装手順  
**対象者**: Phase 3 backtest_daily()実装チーム  
**前提条件**: Phase 1（環境準備）・Phase 2（ドキュメント化）完了済み  

---

## 🎯 実装目的と背景

### 核心的目標
**DSSMS日次判断とマルチ戦略全期間一括判定の設計不一致を解決し、リアルトレード対応の日次バックテストシステムを構築する。**

### 現在の問題（Phase 1-2完了時点）
```python
# 設計不一致の核心
DSSMS側: 毎日最適な1銘柄を選択 → 日次判断前提
マルチ戦略側: backtest(start, end) → 全期間一括判定

# 結果: リアルトレードと乖離したバックテスト
kabu STATION API統合時に設計的な障壁となる
```

### Phase 3で実現する設計
```python
# 目標: 統一された日次対応設計
BaseStrategy.backtest_daily(current_date, stock_data, existing_position)
→ その日のみを判定（リアルトレードと完全一致）
→ 銘柄切替対応・ポジション継続性・ルックアヘッドバイアス防止
```

---

## ⚠️ 重要な注意事項

### 絶対に遵守すべき制約
1. **ルックアヘッドバイアス防止（copilot-instructions.md必須）**:
   ```python
   # 禁止: 当日データで判定
   signal = data['Close'].iloc[current_idx]
   
   # 必須: 前日データで判定 + 翌日始値エントリー
   signal = data['Close'].shift(1).iloc[current_idx]
   entry_price = data['Open'].iloc[current_idx + 1] * (1 + slippage)
   ```

2. **バックテスト実行必須（copilot-instructions.md基本原則）**:
   - 全ての実装で`strategy.backtest()`または`strategy.backtest_daily()`の実際実行必須
   - 実際の取引件数 > 0 を検証必須
   - 推測ではなく正確な数値報告必須

3. **フォールバック機能禁止（copilot-instructions.md制約）**:
   - モック/ダミー/テストデータを使用するフォールバック禁止
   - エラー隠蔽を目的としたフォールバック禁止
   - フォールバック発見時は必ず報告

4. **決定論保証**:
   - 同じ入力に対して常に同じ出力を保証
   - ランダム性の排除（DSSMS決定論モード遵守）

### リスクの高い変更箇所
- **BaseStrategy.py**: 全戦略に影響する基底クラス
- **DSSMS統合部**: 銘柄切替・ポジション継続性に関わる
- **PaperBroker状態管理**: 資金リセット問題再発リスク

---

## 🚀 アジャイル実装ステップ

### Phase 3-A: MVP実装（3-4日）

#### Day 1-2: 基盤構築
**目標**: 動作するMVPシステムの構築

**Step A1: BaseStrategy.backtest_daily()スケルトン実装**
```python
# strategies/base_strategy.py
def backtest_daily(self, current_date: datetime, stock_data: pd.DataFrame, 
                  existing_position: Optional[Dict] = None) -> Dict[str, Any]:
    """
    日次バックテスト実行（MVP版）
    
    Returns:
        {
            'action': 'entry'|'exit'|'hold',
            'signal': 1|-1|0,
            'price': float,
            'shares': int,
            'reason': str
        }
    """
    # Phase 3-A: デフォルト実装（既存backtest()をラップ）
    # 後方互換性を保ちながら新インターフェースを確立
```

**検証ポイント**:
- [ ] 既存戦略が動作し続ける（後方互換性）
- [ ] 新インターフェースが呼び出し可能
- [ ] 基本的なリターン形式が正しい

**Step A2: VWAPBreakout戦略での実証実装**
```python
# strategies/vwap_breakout_strategy.py
def backtest_daily(self, current_date, stock_data, existing_position=None):
    # 実装内容:
    # 1. current_dateのindexを特定
    # 2. ウォームアップ期間考慮（150日）
    # 3. 前日データのみでVWAPインジケーター計算
    # 4. エントリー判定（ルックアヘッドバイアス防止）
    # 5. 翌日始値エントリー価格設定
```

**検証ポイント**:
- [ ] インジケーターに`.shift(1)`適用済み
- [ ] エントリー価格が`data['Open'].iloc[idx + 1]`
- [ ] 実際の取引件数 > 0 を確認
- [ ] 既存backtest()との結果比較

#### Day 3-4: 統合テスト
**Step A3: DSSMS統合の最小実装**
```python
# src/dssms/dssms_integrated_main.py
def _execute_multi_strategies_daily(self, target_date, symbol, stock_data):
    """Phase 3-A MVP版統合実行"""
    # 1戦略（VWAPBreakout）のみでの統合テスト
    strategy = VWAPBreakoutStrategy(stock_data)
    result = strategy.backtest_daily(target_date, stock_data)
    
    # 結果処理・ログ記録
    return result
```

**検証ポイント**:
- [ ] DSSMS → backtest_daily()の呼び出し成功
- [ ] target_dateでの正常な判定実行
- [ ] 銘柄切替なしでの動作確認
- [ ] エラーハンドリング動作確認

### Phase 3-B: パターン確立（2-3日）

#### Day 5-6: 改善とテンプレート化
**目標**: 他戦略展開のためのパターン確立

**Step B1: Phase 3-Aで発見された問題の解決**
- インジケーター計算の最適化
- エラーハンドリングの改善
- パフォーマンス問題の修正

**Step B2: 展開テンプレートの作成**
```python
# templates/backtest_daily_template.py
"""
backtest_daily()実装テンプレート

使用手順:
1. existing_positionハンドリングコピー
2. インジケーター計算部分のshift(1)適用
3. エントリー価格設定部分のOpen価格使用
4. リターン形式の統一
"""
```

**検証ポイント**:
- [ ] テンプレートが他戦略に適用可能
- [ ] コード生成効率の向上確認
- [ ] 品質の一貫性保証確認

#### Day 7: 第2戦略での検証
**Step B3: 2つ目の戦略実装**
- Momentum戦略またはBreakout戦略を選択
- テンプレート活用での実装
- パターンの有効性検証

### Phase 3-C: スケールアウト（5-7日）

#### Day 8-11: 全戦略展開
**目標**: 確立されたパターンでの全戦略実装

**戦略実装優先順序**:
1. **VWAPBreakout** (完了)
2. **Momentum** (Day 8)
3. **Breakout** (Day 9)  
4. **Contrarian** (Day 10)
5. **その他戦略** (Day 11)

**各戦略での検証チェックリスト**:
- [ ] インジケーター計算でshift(1)適用
- [ ] エントリー価格でOpen価格使用
- [ ] 実取引件数 > 0 確認
- [ ] 既存backtest()との一貫性確認
- [ ] エラー発生時の適切なハンドリング

#### Day 12-14: 統合テスト・最適化
**Step C1: 全戦略統合でのDSSMSテスト**
```python
# 統合テストシナリオ
Day 1: 6954選択 → VWAPBreakout実行
Day 2: 9101切替 → ポジション処理 → Momentum実行  
Day 3: 9101継続 → 既存ポジション考慮 → 判定実行
```

**Step C2: パフォーマンス最適化**
- データ取得の効率化
- インジケーター計算の最適化
- メモリ使用量の削減

**Step C3: 銘柄切替対応の実装**
```python
def handle_symbol_switch(self, old_symbol, new_symbol, target_date):
    # 旧銘柄ポジション決済
    # 新銘柄での新規判定
    # 資金・リスク管理継続
```

---

## 🔄 継続的改善サイクル

### 各フェーズでの改善サイクル
```
実装 → テスト → 問題発見 → 修正 → 再テスト
  ↓
パターン改良 → 次戦略適用 → 検証 → テンプレート更新
  ↓  
統合テスト → パフォーマンス測定 → 最適化 → 再統合テスト
```

### 継続的な品質保証
1. **毎日の動作検証**:
   - 実取引件数の確認
   - エラーログの監視
   - パフォーマンス指標の測定

2. **週次のレビューサイクル**:
   - 実装パターンの見直し
   - テンプレートの改善
   - ドキュメント更新

---

## 🧪 テスト戦略

### 単体テスト（戦略レベル）
```python
def test_backtest_daily_lookback_bias():
    """ルックアヘッドバイアス防止テスト"""
    # 未来データ使用の検出
    
def test_backtest_daily_determinism():
    """決定論テスト"""
    # 同じ入力で同じ出力の保証
    
def test_backtest_daily_position_handling():
    """既存ポジション処理テスト"""
    # existing_position引数の適切な処理
```

### 統合テスト（システムレベル）
```python
def test_dssms_daily_integration():
    """DSSMS統合テスト"""
    # 日次判断とマルチ戦略の整合性
    
def test_symbol_switch_handling():
    """銘柄切替テスト"""
    # ポジション継続性とリスク管理
    
def test_real_trade_simulation():
    """リアルトレード模擬テスト"""
    # kabu STATION API呼び出し前の最終検証
```

### パフォーマンステスト
```python
def test_execution_performance():
    """実行時間測定"""
    # 1日あたりの処理時間監視
    
def test_memory_usage():
    """メモリ使用量測定"""
    # 長期間実行でのメモリリーク監視
```

---

## 📈 成功指標

### Phase 3-A成功指標
- [ ] VWAPBreakout戦略でbacktest_daily()が動作
- [ ] 実取引件数 > 0 を確認
- [ ] ルックアヘッドバイアステスト合格
- [ ] DSSMS統合での基本動作確認

### Phase 3-B成功指標  
- [ ] 2つの戦略でbacktest_daily()動作
- [ ] 実装テンプレートの有効性確認
- [ ] 品質の一貫性確保
- [ ] 実装効率の向上確認

### Phase 3-C成功指標
- [ ] 全戦略でbacktest_daily()動作
- [ ] 銘柄切替対応の完全動作
- [ ] パフォーマンス要件達成
- [ ] リアルトレード模擬テスト合格

---

## ⚡ 緊急時対応

### 重大問題発生時の対応手順
1. **問題の即座な隔離**: 該当戦略の無効化
2. **ログの詳細確認**: ERROR/WARNINGレベルの分析
3. **ロールバック判断**: Phase 1環境への復旧検討
4. **チーム報告**: 問題の詳細と影響範囲の共有
5. **修正方針決定**: 緊急修正 vs 再設計判断

### よくある問題と対処法
**問題1**: インジケーター計算でshift(1)忘れ
→ **対処**: ルックアヘッドバイアステストで即座に検出・修正

**問題2**: 銘柄切替時のポジション処理エラー
→ **対処**: existing_positionハンドリングの見直し

**問題3**: パフォーマンス劣化
→ **対処**: データ取得方法の最適化・キャッシュ活用

---

## 📚 参考資料

### 必読ドキュメント
- [DESIGN_DECISION_ANALYSIS_20251230.md](Fund%20position%20reset%20issue/DESIGN_DECISION_ANALYSIS_20251230.md): 設計問題の詳細分析
- [backtest_daily_implementation_guide.md](backtest_daily_implementation_guide.md): 実装ガイド詳細版
- [copilot-instructions.md](../.github/copilot-instructions.md): 開発ルール（ルックアヘッドバイアス禁止等）

### コード参照
- [strategies/base_strategy.py](../strategies/base_strategy.py): 現行backtest()実装
- [src/dssms/dssms_integrated_main.py](../src/dssms/dssms_integrated_main.py): DSSMS統合実装
- [strategies/vwap_breakout_strategy.py](../strategies/vwap_breakout_strategy.py): 最初の実装対象

### 外部参考
- kabu STATION API仕様: リアルトレード制約の理解
- pandas.DataFrame.shift(): ルックアヘッドバイアス防止の基本

---

## 🎉 Phase 3完了後の次のステップ

### Phase 4: 最終整理（予定）
- 旧backtest()メソッドの段階的廃止
- コードベースの整理・最適化
- プロダクション環境準備

### kabu STATION API統合準備
- backtest_daily()をベースとした実トレード実装
- API呼び出しタイミングの最適化
- エラーハンドリング・リスク管理の強化

---

**作成者**: GitHub Copilot  
**Phase**: Phase 2 ドキュメント化 → Phase 3 実装準備  
**予想総工数**: 2週間（Phase 3-A: 4日 + Phase 3-B: 3日 + Phase 3-C: 7日）  
**前提条件**: Phase 1（累積期間復元）・Phase 2（ドキュメント化）完了  
**成功の鍵**: アジャイル的な継続改善とcopilot-instructions.mdルール遵守  

**重要**: このアジャイル実装により、リアルトレード対応の基盤が完成し、kabu STATION API統合への道筋が開かれる。