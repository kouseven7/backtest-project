# DSSMSバックテスト 取引件数異常問題 - 詳細設計書

## 📋 調査結果サマリー

### 根本原因
**yfinanceのend_dateパラメータはexclusive（含まない）として扱われるため、`data_cache_manager.py`の`_fetch_external_data`メソッドで`end_date`をそのまま渡すと、target_dateを含まないデータが返される。**

### 影響
- 2023年1月～12月の1年間バックテストで、**1日分の取引しか発生しない**
- main_new.pyのデータ範囲チェック（Line 197-223）で`trading_start_date > data_last_date`エラーが発生し、361日分の処理が中断

### 検証済み事実
1. yfinance.history()は`end_date`をexclusiveとして扱う（証拠: test_20250314_yfinance_end_date.py）
2. `end_date + timedelta(days=3)`で全エッジケースにおいてtarget_dateを含むデータを取得可能（証拠: test_20251206_yfinance_edge_cases.py）
3. dssms_integrated_main.pyのフォールバック（Line 1859）は実行されない（DataCacheManager.get_cached_dataが内部処理するため）
4. 修正対象は`data_cache_manager.py`の2箇所のみ（Line 489, 498）

---

## 🔧 修正設計

### 修正対象ファイル
**`src/dssms/data_cache_manager.py`**

### 修正箇所1: stock_data取得（Line 489）

#### 修正前
```python
stock_data = ticker.history(start=start_date, end=end_date, auto_adjust=False)
```

#### 修正後
```python
# yfinanceのhistory()はend_dateをexclusiveとして扱うため、
# target_date当日のデータを確実に取得するには+3日の余裕を持たせる
# 理由: 営業日カレンダーの関係で+1日では不十分なケースがある
stock_data = ticker.history(start=start_date, end=end_date + timedelta(days=3), auto_adjust=False)
```

### 修正箇所2: index_data取得（Line 498）

#### 修正前
```python
index_data = nikkei_ticker.history(start=start_date, end=end_date, auto_adjust=False)
```

#### 修正後
```python
# yfinanceのhistory()はend_dateをexclusiveとして扱うため、
# target_date当日のデータを確実に取得するには+3日の余裕を持たせる
index_data = nikkei_ticker.history(start=start_date, end=end_date + timedelta(days=3), auto_adjust=False)
```

### 必要なimport追加
**既に存在**: `from datetime import datetime, timedelta`（Line 11）

---

## ✅ .github/copilot-instructions.md 準拠確認

### フォールバック機能の制限
- **モック/ダミー/テストデータを使用するフォールバック禁止**: ✅ 準拠（yfinanceから実データを取得）
- **テスト継続のみを目的としたフォールバック禁止**: ✅ 準拠（エラー隠蔽ではなく、正しいデータ取得）
- **フォールバック実行時のログ必須**: N/A（フォールバックではない）

### データ取得ルール
- **yfinance auto_adjust=False必須**: ✅ 準拠（既存の`auto_adjust=False`を維持）

---

## 🧪 テスト設計

### テスト1: 単体テスト（修正後のdata_cache_manager.py）

**目的**: `_fetch_external_data`が正しくtarget_dateを含むデータを取得するか検証

**テストケース**:
1. 平日（2023-12-01）
2. 月末（2023-11-30）
3. 月初（2023-12-04）
4. 週末直前（2023-01-27）

**検証項目**:
- stock_dataにtarget_dateが含まれるか
- index_dataにtarget_dateが含まれるか
- Adj Closeカラムが存在するか

**テストファイル**: `tests/temp/test_20251206_data_cache_manager_fix.py`

### テスト2: 統合テスト（DSSMS全体）

**目的**: 1年間バックテストで取引が正常に発生するか検証

**実行コマンド**:
```powershell
python src/dssms/dssms_integrated_main.py --start-date 2023-01-01 --end-date 2023-12-31
```

**検証項目**:
- `output/dssms_trades.csv`に複数の取引が記録されるか（目標: 10件以上）
- `DATA_INSUFFICIENT`エラーが発生しないか
- `portfolio_equity_curve.csv`が正常に生成されるか

**成功基準**:
- 取引件数 > 1件（現状: 1件 → 修正後: 10件以上期待）
- エラー終了: 361日 → 0日

---

## 📊 影響範囲分析

### 修正による影響
1. **キャッシュキー**: 影響なし（キャッシュキー生成はLine 159、データ取得はLine 178で分離）
2. **データ範囲**: 余分なデータ（target_date + 1～3日分）が取得されるが、main_new.pyのフィルタリング（Line 169）で除去される
3. **既存キャッシュ**: 無効化されないが、キャッシュミス時に新しいデータが取得される（TTL 30日で自然に更新）
4. **パフォーマンス**: yfinanceのAPIコール回数は変わらない（キャッシュミス時のみ実行）

### 副作用
**なし** - 安全な修正です。

---

## 🔄 実装手順

### Step 1: バックアップ
```powershell
Copy-Item src\dssms\data_cache_manager.py src\dssms\data_cache_manager.py.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')
```

### Step 2: 修正実施
`multi_replace_string_in_file`を使用して2箇所を同時修正

### Step 3: 単体テスト実行
```powershell
python tests\temp\test_20251206_data_cache_manager_fix.py
```

### Step 4: 統合テスト実行
```powershell
python src\dssms\dssms_integrated_main.py --start-date 2023-01-01 --end-date 2023-02-28
```

### Step 5: 結果検証
- `output/dssms_trades.csv`の取引件数を確認
- `logs/main_system_controller.log`のエラー件数を確認

---

## 📈 期待される改善効果

### 修正前
- 取引件数: **1件**（2023-01-26のみ）
- エラー終了: **361日**
- 成功日: **1日**（0.3%）

### 修正後（予測）
- 取引件数: **5件以上**
- エラー終了: **0日**
- 成功日: **30日以上**

### 理由
- 全ての日付でtarget_dateを含むデータが取得される
- main_new.pyのデータ範囲チェックをパスする
- 戦略が正常にシグナル生成できる

---

## 🚨 注意事項

1. **既存キャッシュのクリアは不要**: TTL（30日）で自然に更新されます
2. **dssms_integrated_main.pyの修正は不要**: DataCacheManagerの修正で解決します
3. **auto_adjust=Falseは維持**: Adj Closeカラムを保証するため

---

## 📝 セルフチェック項目

### 見落としチェック
- [x] data_cache_manager.pyの両方の箇所（stock_data, index_data）を修正
- [x] timedeltaのimportが存在することを確認
- [x] auto_adjust=Falseが維持されることを確認

### 思い込みチェック
- [x] yfinanceのexclusiveルールを実際に検証（test_20250314_yfinance_end_date.py）
- [x] timedelta(days=3)の有効性を実際に検証（test_20251206_yfinance_edge_cases.py）
- [x] DataCacheManagerが実際に使用されることを確認（dssms_integrated_main.py Line 1843）

### 矛盾チェック
- [x] 調査結果に矛盾なし
- [x] copilot-instructions.mdに抵触しない
- [x] 副作用なし

---

## 🎯 完了条件

1. `tests/temp/test_20251206_data_cache_manager_fix.py`が全パス
2. DSSMS統合テストで取引件数 > 1件
3. `DATA_INSUFFICIENT`エラーが0件
4. ログに異常なエラーが記録されない

---

**作成日**: 2025-12-06
**作成者**: GitHub Copilot
**バージョン**: 1.0
