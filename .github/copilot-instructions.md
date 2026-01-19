# Copilot Instructions - 日本語版

🎯 【最優先目的】リアルトレードで利益を上げる
本プロジェクトの最終目的は「リアルトレードで利益を上げること」である。
すべてのコーディング・設計判断は、この目的達成のために行う。
目的達成のための階層
1. バックテストで利益を上げる戦略を構築
   ├─ 正確性の担保（バグ・オーバーフィッティングの排除）
   └─ 実取引件数 > 0 の検証必須
2. ペーパートレードで戦略を検証
3. kabu STATION API経由でリアルトレードへ移行
4. 実利益の獲得
コーディング時の判断基準

✅ この実装は「利益向上」または「バックテストの正確性向上」に寄与するか？
❌ 手段が目的化していないか？（例：保有期間の固定自体が目的になっていないか）
❌ バックテストの信頼性を損なう実装になっていないか？

迷ったら、常にこの目的に立ち返って判断する。

## 📐 **プロジェクトアーキテクチャ**

### **システム階層**
```
1. DSSMS Core (src/dssms/)
   └─ Screening/Ranking/Scoring/Symbol Switching
2. Multi-Strategy Execution (strategies/)
   └─ BaseStrategy派生クラス群（Breakout, Momentum等）
3. Data Layer
   ├─ data_fetcher.py: yfinance統合+CSV cache
   └─ data_cache_manager.py: キャッシュ管理
4. Output & Reporting
   └─ CSV+JSON+TXT統一出力（Excel廃止済み）
```

### **重要な設計思想**
- **DSSMS目的**: 日経225から最適1銘柄を動的選択→マルチ戦略適用→kabu STATION API経由で実トレード
- **バックテスト第一**: すべての戦略は`strategy.backtest()`呼び出しで実際のトレード数・損益を検証
- **決定論モード**: DSSMS切替は完全再現可能（ランダム性排除）
- **分散投資なし**: 単一最適銘柄への集中運用（将来拡張余地のみ確保）

## 🎯 **基本原則**
1. **バックテスト実行必須**: `strategy.backtest()` の呼び出しをスキップしない
2. **検証なしの報告禁止**: 実際の実行結果を確認せず「成功」と報告しない
3. **わからないことは正直に**: 不明な場合は推測せず「わかりません」と回答

## 📋 **品質ルール**
- **報告前に検証**: 実際の実行、実際の数値を確認してから報告
- **Excel出力禁止**: CSV+JSON+TXTを使用（2025-10-08以降）Excel入力は許可する

## ⚠️ **既知の問題**
- Unicode文字はWindowsターミナルでエラーを起こす可能性があるため、2025/10/20より⚠️などの絵文字は使用しないこと

## 🚨 **必須チェック項目**
- 実際の取引件数 > 0 を検証
- 出力ファイルの内容を確認（存在確認だけでは不十分）
- 推測ではなく正確な数値を報告

## 🚫 **フォールバック機能の制限**
- **モック/ダミー/テストデータを使用するフォールバック禁止**: 実データと乖離する結果を生成するフォールバック機能は実装しない
- **テスト継続のみを目的としたフォールバック禁止**: エラーを隠蔽して強制的にテストを継続させるフォールバックは実装しない
- **フォールバック実行時のログ必須**: フォールバック機能が動作した場合は必ずログに記録し、ユーザーが認識できるようにする
- **フォールバックを発見した場合はいかなる場合も報告する**: フォールバックを発見した場合は過去のモジュールであっても通知する

## � **ルックアヘッドバイアス禁止（2025-12-20以降必須）**

### **基本ルール**
```python
# 禁止: 当日終値でエントリー
entry_price = data['Adj Close'].iloc[idx]

# 必須: 翌日始値でエントリー + スリッページ
entry_price = data['Open'].iloc[idx + 1] * (1 + slippage)
```

### **3原則**
1. **前日データで判断**: インジケーターは`.shift(1)`必須
2. **翌日始値でエントリー**: `data['Open'].iloc[idx + 1]`
3. **取引コスト考慮**: スリッページ・を加味

### **対象**
全戦略（BaseStrategy派生クラス全て）

### **発見時の対応**
1. 即座に報告（過去のコードでも）
2. Phase 1（翌日始値）を最優先で修正提案
3. Phase 2（スリッページ等）を推奨として提示

### **チェックリスト**
- [ ] エントリー価格は`data['Open'].iloc[idx + 1]`
- [ ] インジケーターに`.shift(1)`適用
- [ ] スリッページ考慮（推奨0.1%）

## �📝 **コーディング規約**

### **モジュールヘッダーコメント（2025-10-20以降必須）**

すべての新規Pythonモジュール（`.py`ファイル）には、以下の構造を持つヘッダーコメントを**必ず**含めること:

```python
"""
[モジュール名] - [一行での役割説明]

[2-3行での詳細な説明]

主な機能:
- [機能1]
- [機能2]
- [機能3]
- ...（最低3つ、できれば5-7つ）

統合コンポーネント:
- [連携するモジュール1]: [連携内容]
- [連携するモジュール2]: [連携内容]
- ...

セーフティ機能/注意事項:
- [重要な制約や注意点1]
- [重要な制約や注意点2]
- ...

Author: Backtest Project Team
Created: YYYY-MM-DD
Last Modified: YYYY-MM-DD
"""
```

**ヘッダーコメントのチェックリスト:**
- [ ] モジュール名と役割が明確
- [ ] 主な機能が3つ以上リストアップされている
- [ ] 他のモジュールとの連携が明記されている（該当する場合）
- [ ] セーフティ機能や注意事項が記載されている
- [ ] 作成日が記入されている

**例外:**
- `__init__.py`（空の場合）
- テストファイル（`test_*.py`）には簡略版でも可
- 5行未満の極小ユーティリティスクリプト

**違反時の対応:**
新規モジュール作成時にヘッダーコメントが不足している場合、コード生成を**中断**し、ユーザーに以下を確認:
1. ヘッダーコメントを追加するか
2. このモジュールは例外に該当するか

**重要**:このプロジェクトの目的はバックテストの実行とリアルトレードの実行であり、実際のバックテストを妨げる、またはスキップする変更は目的に反します。

## 📊 **データ取得ルール（2025-12-03以降必須）**

### **yfinance auto_adjust=False必須**

すべてのyfinance呼び出しには`auto_adjust=False`を**必ず**指定すること:

**必須理由:**
- yfinanceのデフォルト（`auto_adjust=True`）では`Adj Close`カラムが取得されない（7カラムのみ）
- `auto_adjust=False`指定で8カラム取得され、`Adj Close`が含まれる
- DSSMS等の戦略は`Adj Close`を使用するため、未指定は致命的エラーの原因となる

**対象箇所:**
以下の6箇所は特に重要（2025-12-03修正完了済み）:
1. `data_fetcher.py` Line 181, 229, 242: `yf_download(..., auto_adjust=False)`
2. `data_cache_manager.py` Line 489, 498: `ticker.history(..., auto_adjust=False)`
3. `config/error_handling.py` Line 78: `yf.download(..., auto_adjust=False)`

**コード例:**
```python
# 正しい例
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)

# 誤った例（禁止）
data = yf.download(ticker, start=start_date, end=end_date)  # auto_adjust未指定
stock_data = ticker.history(start=start_date, end=end_date)  # auto_adjust未指定
```

**違反時の影響:**
- CSVキャッシュに`Adj Close`が保存されない
- 補完ロジック（`data['Adj Close'] = data['Close']`）に依存する設計となり、調整後価格の精度が失われる
- 既存キャッシュ（数千ファイル）のクリアと再取得が必要になる

**チェックリスト:**
- [ ] yf.download()呼び出しに`auto_adjust=False`を指定
- [ ] ticker.history()呼び出しに`auto_adjust=False`を指定
- [ ] 取得データに`Adj Close`カラムが含まれることを確認
- [ ] CSVキャッシュ保存時に`Adj Close`が保存されることを確認

**重要**: 新規にyfinanceを使用するコードを作成する際は、必ずこのルールを遵守すること。

## 📁 **テストファイル配置ルール**

### **配置場所の判断**
```
新規テスト作成時
    ↓
繰り返し実行する？
    ├─ YES → tests/core/ (回帰テスト)
    └─ NO  → tests/temp/ (一時テスト、成功後削除)
```

### **一時テスト (tests/temp/)**
- **命名**: `test_YYYYMMDD_<feature>.py`
- **用途**: 新機能の動作確認、一度のみの検証
- **削除基準**:
  - [ ] 全アサーション成功
  - [ ] 実データ検証完了（モック/ダミー不使用）
  - [ ] フォールバックなし動作確認
  - [ ] `docs/test_history/` に記録済み
- **削除**: `python tests/cleanup_temp_tests.py`

### **継続テスト (tests/core/)**
- **用途**: 回帰テスト、CI/CD自動テスト

**重要**: エージェントモードでテストを作成する際は、必ず適切なフォルダに配置すること。

## 🔧 **開発ワークフロー**

### **戦略開発**
```python
# 1. strategies/base_strategy.py を継承
from strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_entry_signal(self, idx):
        # 前日データで判断（.shift(1)必須）
        return signal
    
    def generate_exit_signal(self, idx):
        # エグジット条件
        return signal
    
    def backtest(self, trading_start_date=None, trading_end_date=None):
        # 必ず実装・呼び出し
        return results_df
```

### **実行コマンド**
```powershell
# バックテスト実行
python main.py

# 一時テスト削除
python tests/cleanup_temp_tests.py

# Pytest実行
pytest tests/core/
```

### **データ取得パターン**
```python
# 標準パターン（キャッシュ優先）
from data_fetcher import get_parameters_and_data
ticker, start, end, stock_data, index_data = get_parameters_and_data(
    ticker="9101.T", 
    start_date="2023-01-01", 
    end_date="2024-12-31",
    warmup_days=150  # ウォームアップ期間（Option A-2暦日拡大方式）
)
```

## 🐛 **デバッグTips**

### **戦略でトレードが0件の場合**
1. `generate_entry_signal()`のログを有効化（`DEBUG_BACKTEST=1`環境変数）
2. インジケーターに`.shift(1)`適用済みか確認
3. ウォームアップ期間（150日）が十分か確認
4. トレンドフィルターが厳しすぎないか確認（`detect_unified_trend`）

### **出力ファイルが空の場合**
1. `backtest()`が実際に呼び出されたか確認
2. `results_df`に取引履歴が含まれるか確認
3. 統一出力エンジン呼び出し時のパラメータ確認

## 📊 **重要なファイル**


## 🔗 **統合ポイント**

### **DSSMS ⇔ 戦略層**
- DSSMSは`最適銘柄ticker`を返す → 戦略層は受け取った銘柄でbacktest実行
- 将来実装: リアルタイム切替判定 → kabu STATION API発注

### **戦略 ⇔ データ**
- 戦略は`data_fetcher.get_parameters_and_data()`から`stock_data`を受け取る
- `stock_data`には必ず`Adj Close`列が含まれる（`auto_adjust=False`必須）

### **出力エンジン統合**
- 全戦略の結果は統一出力エンジン経由でCSV+JSON+TXT形式で保存
- Excel依存は完全に廃止済み（入力のみ許可）

## 📋 **Phase実装状況（2026-01-03更新）**

### ✅ **Phase 3-C: マルチ戦略対応拡張（完了率88%）**

**完了日**: 2026年1月3日（Task 6: ドキュメント整備完了）  
**実装範囲**: DynamicStrategySelector統合、マルチ戦略動的選択システム

**主要実装**:
- ✅ MarketAnalyzer統合（市場分析自動実行）
- ✅ DynamicStrategySelector統合（スコアベース戦略選択）
- ✅ ポジション状態管理（existing_position伝達）
- ✅ 動的戦略インスタンス生成（全5戦略対応）
- ✅ Enhanced Logger Manager移行（ログローテーション・圧縮）

**検証完了**:
- ✅ System A統合実行テスト（3/3成功）
- ✅ 全5戦略backtest_daily()実装確認
- ✅ ルックアヘッドバイアス禁止制約準拠（全戦略）

**残課題**: パフォーマンス最適化（データ取得効率化）、overall_status未定義エラー修正

---