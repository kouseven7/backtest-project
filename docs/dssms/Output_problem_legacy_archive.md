# DSSMS 問題アーカイブ（参照用履歴）

**注意**: このファイルは参照用として維持され、将来的な編集は行わないでください。
最新の問題整理と解決計画は `Output problem solving roadmap2.md` を参照してください。

---

## 1. 既存 Problem 群（参照保存版）

### 問題一覧

#### Problem 1: 切替判定ロジックの劣化
- `_evaluate_switch_decision`メソッドは存在するが機能低下
- 24時間最小保有期間制約が過度に厳格
- 週3回以上の切替制限により短期切替が阻害

#### Problem 2: 日付ループ問題
- 2023-12-31 → 2023-01-01 の不正ループが発生
- 日付処理ロジックの不備
- ポートフォリオ価値データフローの追跡必要

#### Problem 3: 決定論的モード副作用
```
enable_score_noise: False
enable_switching_probability: False  
use_fixed_execution: True
```
- 過度な決定論化により切替判定の柔軟性が失われている

#### Problem 4: 統一エンジン混在問題
- DSSMSExcelExporterV2 と 統一エンジン の併用により処理が混乱
- switch_history処理が複数箇所で異なる方法で実行

#### Problem 5: 戦略統計出力不足
- 戦略ごとの統計値が不十分に出力
- 一部指標が欠落

#### Problem 6: データフロー不整合
- 複数の変換処理が混在
- 処理責務の不明確化

#### Problem 7: 保有期間固定問題
- 保有期間が24時間固定される計算ロジック
- 実測値ではなく固定値使用

#### Problem 8: 統計指標未実装
- 勝率/ProfitFactor等の重要指標が未実装
- 出力統計の不足

#### Problem 9: 数式品質格差
- 一部統計数式の品質にばらつき
- 標準化・テスト不足

#### Problem 10: 数学的エラー
- 分母欠如等の基本的計算エラー
- 精度低下要因

#### Problem 11: IntelligentSwitchManager未活用
- 初期化されるが実質未使用
- 切替判断への統合不足

#### Problem 12: 決定論影響（その2）
- 再現性と柔軟性のバランス不足
- 境界値での切替消失

#### Problem 13: マルチエンジン問題
- 複数の類似出力エンジンの並存
- 責務と選択の曖昧さ

#### Problem 14: データ品質軽微問題
- 一部銘柄データの品質問題
- 全体影響は限定的

#### Problem 15: [無効化] トレンド検出偏り
- 誤認識された問題
- 実際は別原因と判明

#### Problem 16: [無効化] ISMログ課題
- 調査の結果、実害なしと判断
- 冗長ログのみの問題

#### Problem 17: エンジン使用不一致問題
**問題**: Task 4.2で85.0点評価された高品質エンジンが使用されていない
- **現状**: `src/dssms/unified_output_engine.py`使用（品質不明）
- **本来**: `dssms_unified_output_engine.py`使用すべき（85.0点確認済み）
- **影響**: 出力品質・切替問題等の根本原因の可能性

#### Problem 18: エンジンファイル整理不備
**問題**: 5個の類似エンジンファイル混在による混乱
- v3空ファイルの放置
- バージョン履歴の無秩序
- 配置場所の標準化不備

### 診断実行結果詳細

#### Task 1.1 切替数カウント機構の検証結果
**確認できた事実**:
1. **初期化状況**：
   - switch_history初期長: 0
   - バックテスター初期化は正常完了
2. **実行パラメータ**：
   - min_holding_period_hours: 24（最小保有期間24時間）
   - 決定論的モード有効（シード=42）
3. **実際の切替実行結果**：
   - テスト期間：2023-01-01～2023-01-10（10日間）

**特定された異常**:
1. **切替頻度の異常な低さ**：
   - 10日間で3回のみ（約3日に1回）
   - 過去の117回から激減している状況が再現
2. **システム設定問題**：
   - `intelligent_switch_manager`が未設定状態
   - 設定ファイル読み込み失敗：市場監視設定でJSONエラー
3. **データ取得警告**：
   - 全銘柄で"possibly delisted"エラー発生
   - これがランキング計算に影響している可能性

#### Task 6.2 エンジンファイル関係性の重大問題発覚
**Task 6.2調査により判明した事実**:
- **Task 4.2最高スコアエンジン**: `dssms_unified_output_engine.py` (85.0点)
- **現在バックテスターが使用**: `unified_output_engine` (srcディレクトリ内)
- **結果**: 85.0点の高品質エンジンが使用されていない状況が確認

**発見されたエンジンファイル（5個の並存問題）**
1. `dssms_unified_output_engine.py` (43,812 bytes, root) - **Task 4.2で85.0点**
2. `dssms_unified_output_engine_fixed.py` (34,675 bytes, root)
3. `dssms_unified_output_engine_fixed_v3.py` (0 bytes, root) - **空ファイル**
4. `dssms_unified_output_engine_fixed_v4.py` (23,725 bytes, root)
5. `src/dssms/unified_output_engine.py` (59,083 bytes, src) - **実際に使用中**

**重大な問題点**
1. **エンジン不一致問題 (Critical)**:
   - 最高品質エンジン(85.0点)が使用されず、品質不明のエンジンを使用
   - バックテスターが`from dssms_unified_output_engine import`と記載されているが、実際は`unified_output_engine`を使用
2. **ファイル整理不備 (High)**:
   - 5個の類似エンジンファイルが混在
   - ルートディレクトリとsrcディレクトリに分散配置
3. **Excel出力責任の混乱 (Medium)**:
   - 複数エンジンにExcel出力機能が分散
   - 実際の出力がどのエンジン由来か不明確

**混乱の根本原因**:
- **バージョン管理の破綻**: fixed → v3 → v4と増殖したが整理されず
- **ファイル配置の無秩序**: root/srcディレクトリの混在
- **使用エンジンの不明確**: インポート文と実際の使用が不一致
- **品質評価との乖離**: 85.0点エンジンが放置状態

---

## 追加 Task 実行結果

### Task 2.1: 日付処理ロジックの検証
**問題点**:
```python
# 年初に発生する無限ループの原因
expected_year = 2023  # ハードコード値
if dt.year != expected_year:
    dt = dt.replace(year=expected_year)
```

### Task 3.1: 保有期間計算ロジックの比較分析
**問題点**:
```python
# 固定24時間使用箇所
holding_period = pd.Timedelta(hours=24)  # 固定値使用
```

### Task 4.1: 統計計算式の検証
**問題点**:
```python
# 分母欠如エラーの例
win_rate = win_count / total_trades  # total_trades=0の時にゼロ割
```

---

## 追加調査項目（将来参照用）

1. **IntelligentSwitchManager統合箇所の特定**
   - 初期化と実使用の差異
   - 必要統合ポイント

2. **データフロー詳細マップ作成**
   - 各変換・処理の詳細追跡
   - 責務分散状況の把握

3. **エンジンファイル詳細比較**
   - 行単位diff
   - 機能差分一覧
   - 最適エンジン選定

---

**注意**: この文書は参照目的のみで維持され、最新の問題解決状況は `Output problem solving roadmap2.md` を参照してください。