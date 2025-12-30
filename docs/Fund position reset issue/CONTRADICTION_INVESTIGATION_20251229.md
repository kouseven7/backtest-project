# main_new.pyとDSSMSの取引件数の矛盾調査レポート

**調査日**: 2025年12月29日  
**調査者**: GitHub Copilot  
**目的**: main_new.pyとDSSMSで同じVWAPBreakoutStrategyを使用しているのに、取引件数が異なる理由を解明する

---

## 1. 調査チェックリスト

### ✅ 確認項目（優先度順）

1. **[完了]** dssms_integrated_main.py Line 1697-1723の詳細確認
2. **[完了]** DSSMSがMainSystemControllerに渡すconfig内容の特定
3. **[完了]** DSSMS→MainSystemController.execute_comprehensive_backtest呼び出しパラメータ確認
4. **[完了]** main_new.pyの実行パラメータ確認（ターミナル履歴から）
5. **[完了]** 銘柄切替（15回/33日）の影響分析
6. **[調査中]** 調査結果まとめと矛盾の解明

---

## 2. 証拠ベースの調査結果

### 2.1 main_new.pyの実行パラメータ（ターミナル履歴より）

**根拠**: ターミナル履歴
```
Last Command: python main_new.py --ticker 6954 --start-date 2024-09-01 --end-date 2025-01-31
Exit Code: 0
```

**main_new.py実行結果**（2024-09-01〜2025-01-31、111行のデータ）:
- **銘柄**: 6954.T（ファナック、固定）
- **取引件数**: 2件
- **取引詳細**:
  - 取引1: 2025-01-15〜2025-01-22（7日、+50,617円、+5.7%）
  - 取引2: 2025-01-24〜2025-01-29（5日、-5,825円、-1.2%）

**重要**: main_new.pyは`2024-09-01〜2025-01-31`の期間でテストしたが、実際の取引は`2025-01-15〜2025-01-31`の期間に発生している。

---

### 2.2 DSSMSの実行パラメータ

**根拠**: dssms_integrated_main.py Line 1728-1747
```python
# 【Option A実装】2025-12-28
# 日次ウォームアップ方式: target_dateのみをバックテスト対象とし、ウォームアップ期間で過去データを使用
backtest_start_date = target_date  # 当日のみ
backtest_end_date = target_date
warmup_days = self.warmup_days  # クラス変数を使用（150日）
```

**DSSMS実行結果**（2025-01-15〜2025-02-28、33営業日）:
- **銘柄**: 動的選択（15回切替、45.5%切替率）
- **取引件数**: 1件
- **取引詳細**:
  - 取引1: 2025-02-03〜2025-02-06（3日、-275円、強制決済）

**重要**: DSSMSは**日次ウォームアップ方式**を採用しており、各target_dateで`backtest_start_date = target_date`としている。

---

### 2.3 MainSystemControllerに渡されるconfig内容の比較

#### **main_new.pyのconfig（Line 493-507）**
```python
config = {
    'execution': {
        'execution_mode': 'simple',
        'broker': {
            'initial_cash': 1000000,
            'commission_per_trade': 1.0
        }
    },
    'risk_management': {
        'use_enhanced_risk': False,
        'max_drawdown_threshold': 0.15
    },
    'performance': {
        'use_aggregator': False
    }
}
```

#### **DSSMSのconfig（Line 1700-1715）**
```python
config = {
    'execution': {
        'execution_mode': 'simple',
        'broker': {
            'initial_cash': self.config.get('initial_capital', 1000000),  # 1,000,000円
            'commission_per_trade': 1.0
        }
    },
    'risk_management': {
        'use_enhanced_risk': False,
        'max_drawdown_threshold': 0.15
    },
    'performance': {
        'use_aggregator': False
    },
    'suppress_report_generation': True  # Phase 2: DSSMS経由の呼び出し時はレポート生成抑制
}
```

**差異**:
- DSSMSは`suppress_report_generation': True`を追加
- それ以外は**完全に同一**

**結論**: config内容の違いは取引件数に影響しない。

---

### 2.4 execute_comprehensive_backtest呼び出しパラメータの比較

#### **main_new.pyの呼び出し（Line 555-561）**
```python
results = system.execute_comprehensive_backtest(
    ticker,
    stock_data=stock_data,
    index_data=index_data,
    backtest_start_date=backtest_start,  # 2024-09-01
    backtest_end_date=backtest_end        # 2025-01-31
)
```

**パラメータ**:
- `ticker`: "6954"（固定）
- `backtest_start_date`: 2024-09-01
- `backtest_end_date`: 2025-01-31
- `warmup_days`: デフォルト150日
- `force_close_on_entry`: デフォルトFalse

#### **DSSMSの呼び出し（Line 1773-1781）**
```python
result = self.main_controller.execute_comprehensive_backtest(
    ticker=symbol,
    stock_data=stock_data,
    index_data=index_data,
    backtest_start_date=backtest_start_date,  # target_date（当日のみ）
    backtest_end_date=backtest_end_date,      # target_date（当日のみ）
    warmup_days=warmup_days,                  # 150日
    force_close_on_entry=force_close_on_entry # 銘柄切替時True
)
```

**パラメータ**:
- `ticker`: 動的選択（日次で変動）
- `backtest_start_date`: **target_date（当日のみ）**
- `backtest_end_date`: **target_date（当日のみ）**
- `warmup_days`: 150日
- `force_close_on_entry`: 銘柄切替時True

---

### 2.5 **重大発見**: バックテスト期間の決定的な違い

#### **main_new.pyのバックテスト期間**
```
backtest_start_date: 2024-09-01
backtest_end_date: 2025-01-31
→ 連続的な期間（約5ヶ月、111行）
```

#### **DSSMSのバックテスト期間**
```
backtest_start_date: target_date（当日）
backtest_end_date: target_date（当日）
→ 日次実行（33日間、1日ずつ実行）
```

**証拠**: dssms_integrated_main.py Line 1728-1730
```python
# 【Option A実装】2025-12-28
# 日次ウォームアップ方式: target_dateのみをバックテスト対象とし、ウォームアップ期間で過去データを使用
backtest_start_date = target_date  # 当日のみ
backtest_end_date = target_date
```

---

### 2.6 銘柄切替の影響分析

**根拠**: 長期テストログおよびINVESTIGATION_REPORT_20251229.md
```
銘柄切替: 15回
取引日数: 33日
切替率: 45.5%（15/33）
```

**分析**:
- 33日中15回の銘柄切替（約45%の日で銘柄が変わる）
- 切替時は`force_close_on_entry=True`で既存ポジションを強制決済
- 切替により、エントリー機会が減少する可能性

**影響**:
- 銘柄が頻繁に変わるため、VWAPBreakoutStrategyの確認バー条件（`confirmation_bars=1`）を満たす機会が減少
- 例: 2025-01-15に銘柄Aでエントリー条件を満たしても、翌日に銘柄Bに切替わるとエントリーできない

---

## 3. 矛盾の解明：決定的な違い

### 3.1 **根本原因**: バックテスト期間の違い

#### **main_new.py**: 連続的な期間でのバックテスト
```
期間: 2024-09-01〜2025-01-31（約5ヶ月）
方式: 1回の連続的なバックテスト実行
データ: 111行（連続データ）
取引件数: 2件
```

#### **DSSMS**: 日次ウォームアップ方式でのバックテスト
```
期間: 2025-01-15〜2025-02-28（33営業日）
方式: 33回の日次バックテスト実行（1日ずつ）
データ: 各日でウォームアップ期間150日 + 当日1日
取引件数: 1件
```

### 3.2 **なぜ日次実行で取引件数が減るのか？**

**理由1: base_strategy.pyのループ範囲**

base_strategy.py Line 258（Phase 1修正後）:
```python
for idx in range(len(self.data) - 1):  # 最後の行を除外
```

**日次実行の場合**:
- データ: ウォームアップ期間150日 + 当日1日 = 151行
- ループ範囲: `range(151 - 1)` = `range(150)` = idx 0〜149
- **当日のインデックス（idx=150）はループに含まれない**

**連続実行の場合（main_new.py）**:
- データ: 2024-09-01〜2025-01-31 = 111行
- ループ範囲: `range(111 - 1)` = `range(110)` = idx 0〜109
- **2025-01-31以外のすべての日がループに含まれる**

**理由2: 日次実行では「当日」の取引判断ができない**

DSSMSの日次ウォームアップ方式では、`target_date`（当日）がループに含まれないため、当日のエントリー判断ができない。

**例**:
- 2025-01-15がtarget_dateの場合
- データ: 2024-08-17（ウォームアップ開始）〜2025-01-15（当日）= 151行
- ループ範囲: idx 0〜149（2024-08-17〜2025-01-14）
- **2025-01-15（idx=150）はループに含まれない**

### 3.3 **重大な設計ミス**: 日次ウォームアップ方式の問題

**問題**:
- DSSMSは「target_date当日のエントリー判断」を目的としているが、base_strategy.pyのループ範囲が`range(len(self.data) - 1)`のため、当日はループに含まれない
- 結果として、DSSMSは「前日までの判断」しか行えず、当日のエントリー機会を逃している

**証拠**:
- dssms_integrated_main.py Line 1728-1730:
  ```python
  backtest_start_date = target_date  # 当日のみ
  backtest_end_date = target_date
  ```
- base_strategy.py Line 258:
  ```python
  for idx in range(len(self.data) - 1):  # 最後の行を除外
  ```

**期待動作**:
- target_date当日のエントリー判断を行う

**実際の動作**:
- target_date前日までの判断しか行われない

---

## 4. セルフチェック

### a) 見落としチェック
- ✅ dssms_integrated_main.py Line 1697-1723を確認した（config、呼び出しパラメータ）
- ✅ main_new.pyの実行パラメータをターミナル履歴から確認した
- ✅ base_strategy.pyのループ範囲を確認した（Line 258）
- ✅ 銘柄切替の頻度を確認した（15回/33日）
- ✅ 日次ウォームアップ方式の設計を確認した

### b) 思い込みチェック
- ✅ **思い込み**: 「DSSMSとmain_new.pyは同じMainSystemControllerを使っているから結果も同じはず」
  - **事実**: 呼び出しパラメータ（特にbacktest_start_date/backtest_end_date）が異なる
  - **証拠**: dssms_integrated_main.py Line 1728-1730、main_new.py Line 555-561
- ✅ **思い込み**: 「日次ウォームアップ方式は当日の判断を行う」
  - **事実**: base_strategy.pyのループ範囲が`range(len(self.data) - 1)`のため、当日はループに含まれない
  - **証拠**: base_strategy.py Line 258

### c) 矛盾チェック
- ✅ **矛盾なし**: 調査結果は整合している
  - main_new.pyは連続的な期間でのバックテスト（111行、取引2件）
  - DSSMSは日次ウォームアップ方式でのバックテスト（33日、各日151行、取引1件）
  - 日次実行では当日がループに含まれないため、取引機会が減少する

---

## 5. 調査結果まとめ

### 判明したこと（証拠付き）

1. **config内容の違いは影響しない**
   - 根拠: dssms_integrated_main.py Line 1700-1715、main_new.py Line 493-507
   - 差異: `suppress_report_generation`のみ（取引件数に影響しない）

2. **バックテスト期間の決定的な違い**
   - 根拠: dssms_integrated_main.py Line 1728-1730、main_new.py Line 555-561
   - main_new.py: 連続的な期間（2024-09-01〜2025-01-31、111行）
   - DSSMS: 日次ウォームアップ方式（各日でtarget_date当日のみ、151行）

3. **日次実行では当日がループに含まれない**
   - 根拠: base_strategy.py Line 258（`range(len(self.data) - 1)`）
   - 影響: target_date当日のエントリー判断ができない

4. **銘柄切替の影響**
   - 根拠: 長期テストログ、INVESTIGATION_REPORT_20251229.md
   - 切替率: 45.5%（15回/33日）
   - 影響: エントリー機会の減少

### 不明な点

1. **なぜbase_strategy.pyは`range(len(self.data) - 1)`なのか？**
   - Phase 1修正時の意図が不明
   - ルックアヘッドバイアス対策で最終行を除外している可能性

2. **日次ウォームアップ方式の設計意図**
   - DSSMSは「target_date当日のエントリー判断」を目的としているはずだが、現状では前日までの判断しか行われていない
   - Option A実装（2025-12-28）の設計意図と実装が乖離している可能性

### 原因の推定（可能性順）

#### **可能性1（最も高い）**: base_strategy.pyのループ範囲の問題
- **推定**: `range(len(self.data) - 1)`により、日次実行時に当日がループに含まれない
- **証拠**: base_strategy.py Line 258
- **影響度**: 極めて高い（当日のエントリー判断ができない）
- **修正案**: ループ範囲を`range(len(self.data))`に変更（ただし、ルックアヘッドバイアスに注意）

#### **可能性2（高い）**: 日次ウォームアップ方式の設計ミス
- **推定**: DSSMSの日次ウォームアップ方式は「target_date当日のエントリー判断」を目的としているが、base_strategy.pyのループ範囲と整合していない
- **証拠**: dssms_integrated_main.py Line 1728-1730（backtest_start_date = target_date）
- **影響度**: 高い（設計意図と実装が乖離）
- **修正案**: 
  - Option A: base_strategy.pyのループ範囲を調整
  - Option B: DSSMSのbacktest_start_dateを`target_date - 1日`に変更

#### **可能性3（中程度）**: 銘柄切替の影響
- **推定**: 33日中15回の銘柄切替（45.5%）により、エントリー機会が減少
- **証拠**: 長期テストログ（銘柄切替15回/33日）
- **影響度**: 中程度（エントリー機会の減少に寄与）
- **修正案**: 銘柄切替のロジックを見直し、切替頻度を低減

---

## 6. 推奨アクション

### 優先度A（必須）: base_strategy.pyのループ範囲の調査

**目的**: なぜ`range(len(self.data) - 1)`なのか、Phase 1修正時の意図を確認

**手順**:
1. Phase 1修正時のコミット履歴確認
2. ルックアヘッドバイアス対策の必要性確認
3. 日次ウォームアップ方式との整合性確認

### 優先度B（重要）: DSSMSの設計意図の確認

**目的**: 日次ウォームアップ方式の設計意図と実装の乖離を解消

**手順**:
1. Option A実装（2025-12-28）の設計意図を確認
2. 「target_date当日のエントリー判断」が可能か検証
3. 必要に応じて修正案を提案

### 優先度C（推奨）: 銘柄切替ロジックの見直し

**目的**: エントリー機会の減少を防ぐ

**手順**:
1. 銘柄切替の頻度を分析（45.5%は高いか？）
2. 切替条件を見直し
3. エントリー機会と切替頻度のトレードオフを評価

---

## 7. 結論

**main_new.pyとDSSMSで取引件数が異なる根本原因は、バックテスト期間の違いとbase_strategy.pyのループ範囲の問題である。**

**具体的には**:
1. main_new.pyは連続的な期間（2024-09-01〜2025-01-31）でバックテストを実行し、2件の取引が発生
2. DSSMSは日次ウォームアップ方式で33日間を1日ずつバックテストするが、base_strategy.pyのループ範囲が`range(len(self.data) - 1)`のため、各日の「当日」がループに含まれず、エントリー判断ができない
3. 結果として、DSSMSでは取引機会が大幅に減少し、1件のみとなった

**修正の必要性**:
- base_strategy.pyのループ範囲の調整（ただし、ルックアヘッドバイアスに注意）
- DSSMSの日次ウォームアップ方式の設計見直し

---

**調査完了日**: 2025年12月29日  
**次のステップ**: 優先度A（base_strategy.pyのループ範囲の調査）を実施
