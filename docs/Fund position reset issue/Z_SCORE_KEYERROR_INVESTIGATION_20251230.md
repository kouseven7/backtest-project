"""
Critical Issue調査: Z_Score KeyError問題の詳細分析

作成日: 2025-12-30
目的: MeanReversionStrategy.backtest_daily()でのZ_Score KeyError原因調査
調査対象: 単日実行時のインジケーター計算問題

## 問題の要約
- backtest_daily() 実行時に `KeyError: 'Z_Score'` が発生
- 既存backtest() は正常動作（取引件数2件確認済み）
- 問題はMeanReversionStrategy使用時に発生

## 調査結果

### 1. Z_Score生成ロジック（確認済み）
**ファイル**: strategies/mean_reversion_strategy.py, Lines 89-97
**生成箇所**: initialize_strategy() メソッド内
**計算方式**: 
```python
z_sma = self.data[price_column].rolling(window=zscore_period).mean()
z_std = self.data[price_column].rolling(window=zscore_period).std()
self.data['Z_Score'] = ((self.data[price_column] - z_sma) / z_std).shift(1)
```

**特徴**:
- ルックアヘッドバイアス防止のため `.shift(1)` 適用済み
- zscore_period=15日のローリング計算
- initialize_strategy()はコンストラクタから呼び出される

### 2. Z_Score使用箇所（確認済み）
**エラー発生箇所**: strategies/mean_reversion_strategy.py, Line 145
```python
z_score_val = self.data['Z_Score'].iloc[idx]
```

**使用箇所**:
1. generate_entry_signal() 内（Line 145）- **エラー発生箇所**
2. generate_exit_signal() 内（Line 261、安全チェック付き）

### 3. 実行パス分析

#### 既存backtest()の成功パス:
```
MeanReversionStrategy.__init__() 
→ BaseStrategy.__init__(data, params) 
→ initialize_strategy() 
→ Z_Scoreカラム生成
→ backtest()実行
→ generate_entry_signal()でZ_Score参照 ✅成功
```

#### backtest_daily()の失敗パス（推定）:
```
MeanReversionStrategy.__init__() 
→ BaseStrategy.__init__(data, params)
→ initialize_strategy()  
→ Z_Scoreカラム生成
→ backtest_daily()実行
→ BaseStrategy.backtest()ラッパー呼び出し
→ ??? (データ操作・コピー?)
→ generate_entry_signal()でZ_Score参照 ❌KeyError
```

## 重要な発見

### BaseStrategy.backtest_daily()のデータ操作疑惑
**ファイル**: strategies/base_strategy.py, Lines 421-424
```python
# 一時的にself.dataを更新（既存backtest()との互換性のため）
original_data = self.data
self.data = stock_data.copy()
```

**問題の可能性**: 
- `self.data = stock_data.copy()` によりZ_Scoreカラムが消失
- stock_dataは外部からの生データ（基本6カラム: OHLCV + Adj Close）
- initialize_strategy()で生成されたZ_Scoreカラムが含まれない

### データ復元処理
**ファイル**: strategies/base_strategy.py, Lines 515-517
```python
finally:
    if 'original_data' in locals():
        self.data = original_data
```

## 推定される根本原因

**仮説1（最も可能性が高い）**: データ置き換え問題
- backtest_daily()が stock_data.copy() で self.data を置き換える
- stock_data には Z_Score カラムが含まれない（生データのみ）
- generate_entry_signal() でアクセス時にKeyError発生

**仮説2**: 単日期間での計算不足
- 単日実行時のローリング計算（15日期間）で十分なデータ不足
- ただし、ウォームアップ期間150日が設定されているため可能性低い

**仮説3**: コピー処理のタイミング問題
- stock_data.copy() 時点で必要なカラムが適切にコピーされない
- インデックス不整合やカラム名不一致

## 証拠と確認事項

### ✅ 確認済み事項:
- Z_Score計算ロジックは正常に実装
- 既存backtest()は正常動作（取引件数2件）
- ルックアヘッドバイアス防止は適切に実装
- エラー発生箇所の特定

### ❓ 要確認事項:
- backtest_daily()実行時の self.data の実際の内容
- stock_data に含まれるカラムの確認
- BaseStrategy.backtest_daily() でのデータフロー詳細
- other戦略での同様問題の有無

## 推奨する次の調査ステップ

1. **データ内容の実証確認**: backtest_daily()実行前後でのself.dataカラム確認
2. **stock_dataの内容確認**: 渡されるデータの構造確認
3. **修正方針の検討**: データ置き換え方法の改良案検討

## copilot-instructions.md遵守状況
- ✅ 実データのみ使用（推測と事実を区別）
- ✅ 実際の取引件数確認済み（既存backtest: 2件）
- ✅ フォールバック機能の回避
- ✅ バックテスト実行必須の確認

Author: GitHub Copilot
Created: 2025-12-30
Status: 調査中（修正前段階）
"""