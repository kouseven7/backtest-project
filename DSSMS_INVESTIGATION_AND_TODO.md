# DSSMS統合バックテスト 調査結果と今後のタスク

**作成日:** 2025-12-07  
**調査期間:** 2023-01-01 ~ 2023-01-31 (22取引日)  
**ステータス:** 調査完了・修正計画策定中

---

## 📊 Executive Summary

### ✅ 完了した修正
1. **DSS Core V3 インポートパス修正（修正案A）**
   - Line 73, 190: `from src.dssms.dssms_backtester_v3 import DSSBacktesterV3`
   - 直接実行・モジュール実行の両方でDSS Core V3が初期化成功
   - **結果:** インポートエラー解消

2. **yfinance auto_adjust=False追加**
   - dssms_backtester_v3.py Line 247: `auto_adjust=False`
   - copilot-instructions.md準拠（2025-12-03以降必須）

### 🔍 調査完了事項

#### 修正前（ユーザー報告）
- **直接実行:** 8 trades, 52.19% return
- **モジュール実行:** 1 trade, 1.95% return
- **原因:** DSS Core V3のインポート失敗（モジュール実行時）

#### 修正後（実測値）

**【直接実行】** `python src/dssms/dssms_integrated_main.py --start-date 2023-01-01 --end-date 2023-01-31`
- **出力フォルダ:** `output/dssms_integration/dssms_20251207_003645/` (1フォルダ)
- **取引件数:** 4件
- **総収益率:** 27.53%
- **勝率:** 75.00% (3勝1敗)
- **最終資本:** 1,275,345円
- **銘柄切替:** 8回 (8316, 8306, 8411, 6758, 8001)
- **DSS Core V3:** 初期化成功
- **パーフェクトオーダー計算:** 50銘柄処理

**【モジュール実行】** `python -m src.dssms.dssms_integrated_main --start-date 2023-01-01 --end-date 2023-01-31`
- **出力フォルダ:** 2フォルダに分散
  - `dssms_20251207_100629/` (2ファイル: switch_history, comprehensive_report)
  - `dssms_20251207_100630/` (8ファイル: その他すべて)
- **取引件数:** 5件（✅ 修正前1件から改善）
- **総収益率:** 32.53%（✅ 修正前1.95%から改善）
- **勝率:** 80.00% (4勝1敗)
- **最終資本:** 1,325,315円
- **銘柄切替:** 8回（✅ 修正前1回から改善）
- **DSS Core V3:** 初期化成功＋動的選択も機能（✅ 完全動作）

---

## 🚨 発見された問題

### 【優先度: 最高】未解決の重大問題

#### 1. ~~モジュール実行時の動的銘柄選択が機能していない~~ ✅ 解決済み
**修正案Aにより解決:**
- 修正前: 1回のみ（初期選択のみ）
- 修正後: 8回の銘柄切替（直接実行と同等）
- 取引件数: 1件 → 5件に改善
- 収益率: 1.95% → 32.53%に改善
- **DSS Core V3の動的選択機能が完全に動作**

**残存する疑問:**
- 直接実行は4件、モジュール実行は5件と微妙に異なる理由（要調査）

#### 2. ファイル出力が2フォルダに分散（修正前・修正後とも継続）
**証拠:**
- 修正前: dssms_20251207_004818 + dssms_20251207_004819
- 修正後: dssms_20251207_100629 + dssms_20251207_100630
- 直接実行は1フォルダのみ（dssms_20251207_003645）
- タイムスタンプが1秒ずれている

**影響:**
- ファイルの所在が不明確
- レポート生成の信頼性への懸念

**原因推定:**
- DSSMSReportGenerator（comprehensive_report, switch_history）
- ComprehensiveReporter（その他ファイル）
- 2つのレポートシステムが異なるタイムスタンプで出力ディレクトリを生成

#### 3. パフォーマンス指標の不一致（直接実行のみ）
**証拠:**
- dssms_comprehensive_report.json: 総収益率52.16%, 最終資本1,522,050円
- dssms_performance_summary.csv: 総収益率27.53%, 最終資本1,275,345円
- 差異: 約25%の収益率差、約25万円の資本差

**影響:**
- レポートの信頼性低下
- ユーザーがどちらを信じるべきか不明

**原因推定:**
- 2つのレポートシステムが異なるデータソースを参照
- equity_curve再構築ロジックの不一致
- ForceCloseトレードの扱いが異なる

### 【優先度: 高】

#### 4. BUY/SELLペアの不一致（直接実行）
**証拠:**
- BUY: 5件
- SELL: 6件（うち1件ForceClose）
- ログ: `[PAIRING_MISMATCH] BUY/SELLペア不一致`

**影響:**
- トレード数カウントの曖昧性
- パフォーマンス計算への影響

#### 5. Unicode emoji違反（22箇所）
**場所:** `dssms_backtester_v3.py`
**違反内容:**
- ✓ (checkmark): 9箇所
- ⚠ (warning): 6箇所
- 🏆 (trophy): 2箇所
- 💥 (explosion): 4箇所
- ✨ (sparkle): 1箇所

**影響:**
- ログ出力時のUnicodeEncodeError
- copilot-instructions.md違反（2025-10-20以降禁止）

**実害:**
- エラーログの汚染
- 実行は継続するが、デバッグが困難

---

## 📋 優先度別タスクリスト

### 🔴 優先度: 最高（即座に対応必要）

#### Task 1: ~~モジュール実行時の動的銘柄選択停止原因調査~~ ✅ 不要（修正案Aで解決済み）
**ステータス:** 完了（修正案Aにより解決）

**検証結果:**
- ✅ モジュール実行: 8回の銘柄切替を確認
- ✅ 取引件数: 5件（直接実行4件より多い）
- ✅ DSS Core V3の動的選択が完全に機能

**新たな疑問:**
- 直接実行4件 vs モジュール実行5件の違いは何か？（優先度: 低）

---

#### Task 2: モジュール実行時のファイル出力2フォルダ分散問題修正
**工数見積:** 1-2時間

**調査項目:**
1. タイムスタンプ生成箇所の特定
   - `_generate_outputs()` の呼び出し回数
   - ComprehensiveReporter vs DSSMSReportGenerator
2. 出力ディレクトリ指定の統一
   - `output_dir` パラメータの追跡
   - レポート生成タイミングの同期

**必要なファイル:**
- src/dssms/dssms_integrated_main.py (Line 2690-2794: `_generate_outputs()`)
- main_system/reporting/comprehensive_reporter.py
- src/dssms/report_generator.py

**成功基準:**
- すべてのファイルが1つのフォルダに出力される
- タイムスタンプが統一される

---

#### Task 3: パフォーマンス指標不一致の原因調査と修正
**工数見積:** 2-3時間

**調査項目:**
1. 2つのレポートシステムのデータソース確認
   - dssms_comprehensive_report.json: データ元追跡
   - dssms_performance_summary.csv: データ元追跡
2. equity_curve再構築ロジックの比較
   - `_rebuild_equity_curve()` の詳細確認
3. ForceCloseトレードの扱い
   - 各レポートでのカウント方法
   - PnL計算への影響

**必要なファイル:**
- src/dssms/dssms_integrated_main.py (Line 2424-2488: `_rebuild_equity_curve()`)
- src/dssms/report_generator.py
- main_system/reporting/comprehensive_reporter.py
- src/dssms/performance_metrics.py

**成功基準:**
- 2つのレポートの収益率が一致する
- データソースが明確に文書化される

---

### 🟡 優先度: 高（早期対応推奨）

#### Task 4: BUY/SELLペア不一致の原因調査
**工数見積:** 1時間

**調査項目:**
1. ForceCloseロジックの動作確認
   - 発動条件
   - 通常SELLとの違い
2. ペアリングロジックの確認
   - FIFOルールの適用状況
   - 強制決済の扱い

**必要なファイル:**
- main_system/reporting/main_text_reporter.py
- execution_detailsのペアリング処理箇所

**成功基準:**
- BUY/SELLペアが一致するか、不一致の理由が明確になる

---

#### Task 5: Unicode emoji修正（22箇所）
**工数見積:** 30分

**対象ファイル:** `src/dssms/dssms_backtester_v3.py`

**修正内容:**
```python
# 修正前
self.logger.info(f"✓ {symbol}: {len(data)}日分のデータ取得成功")

# 修正後
self.logger.info(f"[OK] {symbol}: {len(data)}日分のデータ取得成功")
```

**置換対象:**
- ✓ → [OK]
- ⚠ → [WARNING]
- 🏆 → [TOP]
- 💥 → [ERROR]
- ✨ → [SUCCESS]

**成功基準:**
- UnicodeEncodeErrorが発生しない
- ログが正常に出力される

---

### 🟢 優先度: 中（時間があれば対応）

#### Task 6: portfolio_equity_curve.csvの詳細検証
**工数見積:** 1時間

**調査項目:**
- エクイティカーブの連続性
- ドローダウン計算の妥当性
- daily_pnl計算の正確性

---

#### Task 7: 修正前後の完全比較テスト
**工数見積:** 2時間

**前提条件:**
- Task 1, 2, 3の完了

**実施内容:**
1. 修正前の状態を再現（修正をロールバック）
2. 同一期間（2023-01-01 ~ 2023-01-31）でテスト
3. 直接実行・モジュール実行の両方を記録
4. 修正後との詳細比較

---

## 🔧 修正済み事項（記録）

### 修正案A: インポートパス修正（完了）
**日時:** 2025-12-07  
**修正箇所:**
1. `dssms_integrated_main.py` Line 73
2. `dssms_integrated_main.py` Line 190
3. `dssms_backtester_v3.py` Line 247

**修正内容:**
```python
# 修正前
import dssms_backtester_v3

# 修正後
from src.dssms.dssms_backtester_v3 import DSSBacktesterV3
```

**検証結果:**
- ✅ 直接実行: DSS Core V3初期化成功＋動的選択機能
- ✅ モジュール実行: DSS Core V3初期化成功＋動的選択機能
- ✅ **修正案A完全成功: 両実行方法で8回の銘柄切替を確認**

---

## 📈 パフォーマンス比較表

| 実行方法 | 取引件数 | 総収益率 | 勝率 | 最終資本 | 銘柄切替回数 | DSS V3 |
|---------|---------|---------|-----|---------|------------|--------|
| 修正前（直接） | 8件 | 52.19% | ? | ? | ? | ❌ |
| 修正前（モジュール） | 1件 | 1.95% | 100% | 1,019,460円 | 1回 | ❌ |
| **修正後（直接）** | **4件** | **27.53%** | **75%** | **1,275,345円** | **8回** | **✅** |
| **修正後（モジュール）** | **5件** | **32.53%** | **80%** | **1,325,315円** | **8回** | **✅** |

**注意:** 修正案Aは成功。モジュール実行でも動的選択が完全に機能している。

---

## 🎯 次回作業の推奨順序

### Phase 1: 緊急対応（即日～2日以内）
1. ~~**Task 1:** モジュール実行時の動的銘柄選択停止原因調査~~ ✅ 解決済み
2. **Task 2:** ファイル出力2フォルダ分散問題修正（最優先）
3. **Task 5:** Unicode emoji修正（簡単なので早めに完了）

### Phase 2: パフォーマンス改善（3-5日以内）
4. **Task 3:** パフォーマンス指標不一致の修正
5. **Task 4:** BUY/SELLペア不一致の調査

### Phase 3: 検証・最適化（1週間以内）
6. **Task 6:** portfolio_equity_curve.csv検証
7. **Task 7:** 修正前後の完全比較テスト

---

## 📝 作業時のチェックリスト

### 各タスク開始時
- [ ] タスク番号と目的を明確にする
- [ ] 必要なファイルをすべて特定する
- [ ] 工数見積を確認し、分割が必要か判断

### 調査実施時
- [ ] 実際のコードを読む（推測しない）
- [ ] ログ出力で動作を確認する
- [ ] 証拠（ファイルパス、行番号、実際の値）を記録する

### タスク完了時
- [ ] 成功基準をすべて満たしたか確認
- [ ] 副作用がないかチェック
- [ ] このmdファイルを更新（完了日、結果を記録）

---

## 🔍 追加調査が必要な項目

### 不明点1: 修正前の取引件数の違いの原因
- 修正前（直接）: 8件
- 修正後（直接）: 4件
- **疑問:** 修正によって取引件数が減った？それとも期間が異なる？

**調査方法:**
- 修正をロールバックして同一期間でテスト
- log1.txt, log2.txtの復元または再現

### 不明点2: dssms_comprehensive_report.jsonの収益率52.16%の根拠
- csvレポートは27.53%
- **疑問:** どのデータから52.16%が算出されているのか？

**調査方法:**
- DSSMSReportGeneratorのソースコード確認
- daily_results vs execution_details の違い

---

## 📚 参考ファイル

### 修正済みファイル
- `src/dssms/dssms_integrated_main.py`
- `src/dssms/dssms_backtester_v3.py`

### 調査対象ファイル
- `src/dssms/symbol_switch_manager*.py`
- `main_system/reporting/comprehensive_reporter.py`
- `src/dssms/report_generator.py`
- `src/dssms/performance_metrics.py`

### 出力例
- `output/dssms_integration/dssms_20251207_003645/` (直接実行)
- `output/dssms_integration/dssms_20251207_004818/` (モジュール実行 1/2)
- `output/dssms_integration/dssms_20251207_004819/` (モジュール実行 2/2)

---

**最終更新:** 2025-12-07  
**次回更新予定:** Task 1完了時
