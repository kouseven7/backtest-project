# DSSMS切替数激減問題 根本原因解析と解決ロードマップ

## 診断実行結果概要（2025-09-11実行）

### Task 1.1 切替数カウント機構の検証結果

#### [OK] 確認できた事実
1. **初期化状況**：
   - switch_history初期長: 0
   - performance_history初期長: 4
   - バックテスター初期化は正常完了

2. **実行パラメータ**：
   - min_holding_period_hours: 24（最小保有期間24時間）
   - switch_cost_rate: 0.001（切替コスト0.1%）
   - 決定論的モード有効（シード=42）

3. **実際の切替実行結果**：
   - テスト期間：2023-01-01～2023-01-10（10日間）
   - 検出された切替数：**3回のみ**
   - 切替内容：
     - 1回目: CASH -> 9984 (2023-01-01)
     - 2回目: 9984 -> 6758 (2023-01-02) 
     - 3回目: 6758 -> 9984 (2023-01-09)

#### [WARNING] 特定された異常
1. **切替頻度の異常な低さ**：
   - 10日間で3回のみ（約3日に1回）
   - 過去の117回から激減している状況が再現

2. **システム設定問題**：
   - `intelligent_switch_manager`が未設定状態
   - 設定ファイル読み込み失敗：市場監視設定でJSONエラー

3. **データ取得警告**：
   - 全銘柄で"possibly delisted"エラー発生
   - これがランキング計算に影響している可能性

## 根本原因分析

### [TARGET] 主原因候補

#### 1. **切替判定ロジックの劣化**
- `_evaluate_switch_decision`メソッドは存在するが機能低下
- `intelligent_switch_manager`未設定により高度な切替判定が無効

#### 2. **データ品質問題**  
- 銘柄データ取得失敗（404エラー）によりランキング精度が低下
- 不正確なスコアリングが切替機会を減らしている

#### 3. **決定論的モード副作用**
```
enable_score_noise: False
enable_switching_probability: False  
use_fixed_execution: True
```
- 過度な決定論化により切替判定の柔軟性が失われている

#### 4. **最小保有期間制約**
- 24時間の最小保有期間が切替を抑制
- 短期の有利な切替機会を逃している可能性

## 解決ロードマップ

### Phase 1: 緊急対応（即座実行）

#### Task 1.4: データ取得問題の解決
```powershell
# 銘柄データ取得状況の詳細調査
python -c "import yfinance as yf; print(yf.download('7203.T', period='10d').head())"
```

#### Task 1.5: 設定ファイル修正
- `config\dssms\intelligent_switch_config.json`のJSONエラー修正
- 市場監視設定の復旧

### Phase 2: 切替ロジック復旧（1-2日）

#### Task 2.1: intelligent_switch_manager再有効化
- DSSMSBacktester初期化時の設定確認
- IntelligentSwitchManagerの適切な統合

#### Task 2.2: 切替判定パラメータ調整
```python
# 最小保有期間の短縮テスト
min_holding_period_hours: 12  # 24→12時間
switch_cost_rate: 0.0005      # 0.001→0.0005
```

#### Task 2.3: 決定論的モード調整
```python
# 適度なランダム性導入
enable_score_noise: True
noise_factor: 0.05
enable_switching_probability: True
```

### Phase 3: 検証・最適化（2-3日）

#### Task 3.1: 比較テスト実行
```bash
# 修正前後の切替数比較
python diagnose_switch_count_task1.py  # 現状
python main.py --dssms-test-mode      # 修正後
```

#### Task 3.2: 履歴データとの比較
- cb844e0コミット時点での切替数（117回）との比較
- 期間・銘柄・パラメータを統一した検証

#### Task 3.3: パフォーマンス影響評価
- 切替数増加による収益性への影響測定
- リスク調整リターンの改善確認

## 技術的検証項目

### コード調査優先順位
1. `src.dssms.dssms_backtester.py`の`_evaluate_switch_decision`
2. `IntelligentSwitchManager`の設定・初期化プロセス
3. ランキングシステムの精度検証
4. 市場状況判定ロジックの動作確認

### 設定ファイル修正対象
1. `config\dssms\intelligent_switch_config.json`（JSONエラー修正）
2. 市場監視設定ファイルの復旧
3. 切替判定パラメータの最適化

## 期待される改善効果

### 短期目標（1週間以内）
- 切替数を10-20回/期間に回復
- データ取得エラーの解消
- 設定ファイルエラーの解決

### 中期目標（2週間以内）  
- cb844e0レベル（117回）の切替頻度復旧
- 統計計算機能の正常化
- Excelレポート出力の修正

### 長期目標（1ヶ月以内）
- 最適な切替頻度・パラメータの確立
- 堅牢なエラーハンドリング体制
- 包括的な監視・アラート体制

## 次回実行予定アクション

1. **Task 1.4実行**: データ取得問題の詳細調査
2. **設定ファイル修正**: JSONエラーの即座解決  
3. **Task 2.1実行**: IntelligentSwitchManagerの再統合
4. **比較テスト**: 修正効果の定量的検証

---
**作成日時**: 2025-09-11 20:33  
**診断ベース**: diagnose_switch_count_task1.py実行結果  
**状態**: Phase 1準備完了、Task 1.4実行待ち
