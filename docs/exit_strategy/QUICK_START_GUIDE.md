# エグジット戦略開発ガイド - クイックスタート

**作成日**: 2026年1月22日  
**目的**: Phase 3-6失敗を踏まえた新規エグジット戦略開発の開始手順  
**対象読者**: バックテスト実装者、戦略開発者

---

## 📋 このドキュメントセット

### 1. EXIT_STRATEGY_REDESIGN_V2.md（メインドキュメント）

**内容**:
- Phase 3-6失敗の総括（PF=121.07が-98.9%崩壊）
- カーブフィッティング回避の7原則
- 段階的検証プロトコル（Phase 1-3）
- 再利用可能なスクリプト
- 推奨パラメータ空間
- 実装例

**使用タイミング**: 新規エグジット戦略開発開始時に必読

---

### 2. validate_exit_simple_v2.py（検証スクリプト）

**機能**:
- Phase 1: シンプルルール検証（固定パラメータ、10銘柄、5年データ）
- Phase 2: グリッドサーチ（48組み合わせ、TOP 3選定）
- Phase 3: Out-of-Sample検証（汎化性能確認）

**使用方法**:
```bash
# Phase 1実行
python scripts/validate_exit_simple_v2.py --phase 1

# Phase 2実行
python scripts/validate_exit_simple_v2.py --phase 2

# Phase 3実行
python scripts/validate_exit_simple_v2.py --phase 3
```

---

### 3. PHASE6_PARAMETER_REFITTING_ANALYSIS.md（失敗分析）

**内容**:
- Phase 3-6の詳細失敗分析
- カーブフィッティングの証拠
- Option A実施記録（Step 1-3）

**使用タイミング**: 「なぜ Phase 3-6が失敗したのか」を理解したいとき

---

### 4. EXIT_STRATEGY_VALIDATION_REPORT.md（Phase 3レポート）

**内容**:
- 全14エグジット戦略比較表
- TrendFollowing(1/30/0.4): PF=121.07（Phase 3最優秀、後に崩壊）
- カテゴリー別詳細分析

**使用タイミング**: Phase 3でどのパラメータが選ばれたかを確認したいとき

---

## 🚀 開始手順（3ステップ）

### Step 1: EXIT_STRATEGY_REDESIGN_V2.md を読む

**必読セクション**:
1. Phase 3-6失敗の総括（なぜPF=121.07が崩壊したか）
2. カーブフィッティング回避の7原則（最重要）
3. シンプルエグジット戦略の定義（Phase 1-3の目的）

**所要時間**: 30分

---

### Step 2: validate_exit_simple_v2.py で Phase 1実行

**コマンド**:
```bash
python scripts/validate_exit_simple_v2.py --phase 1
```

**実行内容**:
- 10銘柄（業種分散）× 5年データ（2020-2024）
- 固定パラメータ: 損切5%、トレーリング10%、利確なし
- 目標: 平均PF > 1.0、10銘柄中8銘柄でPF > 1.0

**実行時間**: 約10-30分（10銘柄×5年データ取得 + バックテスト）

**結果判定**:
- ✅ PASS（平均PF > 1.0、合格率 >= 80%） → Phase 2へ進む
- ❌ FAIL → パラメータ微調整（損切3-7%、トレーリング5-15%で再試行）

---

### Step 3: 結果に応じて次のステップ

**Phase 1成功の場合**:
```bash
# Phase 2実行（グリッドサーチ、48組み合わせ）
python scripts/validate_exit_simple_v2.py --phase 2
```

**Phase 1失敗の場合**:
- `results/phase1_simple_YYYYMMDD_HHMMSS.csv` を確認
- 失敗銘柄のPF、Win Rate、取引数を分析
- パラメータ微調整（例: 損切7%、トレーリング15%で再試行）

---

## ⚠️ よくある質問

### Q1: なぜ10銘柄必須なのか？

**A**: Phase 3-6の失敗原因の1つが「単一銘柄（7203.T）最適化」。10銘柄で検証することで銘柄依存を排除し、汎化性能を確保。

---

### Q2: なぜPF上限3.0なのか？

**A**: Phase 3でPF=121.07が2025年で-98.9%崩壊した教訓。PF > 3.0は過学習の強い兆候。

---

### Q3: なぜ5年データなのか？

**A**: Phase 3-6の失敗原因の1つが「短期データ（2022-2024、2年間）最適化」。5年データで検証することで市場環境依存を排除。

---

### Q4: Phase 1で失敗したらどうする？

**A**: パラメータ微調整を推奨。以下の範囲で再試行:
- 損切: 3-7%
- トレーリング: 5-15%
- 利確: なし（トレンドを追う）

---

### Q5: Phase 2のグリッドサーチ実行時間は？

**A**: 約30-60分（10銘柄 × 48パラメータ = 480検証）。

---

### Q6: Phase 3のOut-of-Sample検証は必須？

**A**: 必須。In-Sampleのみの評価は過学習を検出不可。Phase 3で劣化率 < 30%を確認してからリアルトレード移行判断。

---

## 📊 成功基準（Phase 1-3）

### Phase 1（シンプルルール検証）

- [ ] 平均PF > 1.0
- [ ] 10銘柄中8銘柄でPF > 1.0
- [ ] 平均取引数 ≥ 20

---

### Phase 2（グリッドサーチ）

- [ ] TOP 3パラメータ選定完了
- [ ] 平均PF > 2.0
- [ ] 平均Win Rate > 40%
- [ ] 最高PF ≤ 3.0（過学習排除）
- [ ] PF標準偏差 < 0.5（銘柄間安定性）

---

### Phase 3（Out-of-Sample検証）

- [ ] TOP 3中2パラメータ以上で劣化率 < 30%
- [ ] Out-of-Sample PF ≥ In-Sample PF × 0.7
- [ ] 推奨パラメータ確定

---

## 🔧 トラブルシューティング

### 問題1: スクリプトが見つからない

**症状**: `python scripts/validate_exit_simple_v2.py --phase 1` でエラー

**対策**: 絶対パスで実行
```bash
python c:\Users\imega\Documents\my_backtest_project\scripts\validate_exit_simple_v2.py --phase 1
```

---

### 問題2: データ取得エラー

**症状**: `yfinance` でデータ取得失敗

**対策**:
1. インターネット接続確認
2. CSVキャッシュ確認（`data_cache/` ディレクトリ）
3. 手動データ取得テスト:
```python
import yfinance as yf
data = yf.download("7203.T", start="2020-01-01", end="2024-12-31", auto_adjust=False)
print(data.head())
```

---

### 問題3: Phase 1実行時間が長い

**症状**: 10銘柄×5年データ取得で30分以上かかる

**対策**:
1. CSVキャッシュを活用（2回目以降は高速化）
2. Quick版実行（3銘柄のみ）:
```python
# validate_exit_simple_v2.py内のVALIDATION_TICKERSを修正
VALIDATION_TICKERS = [
    "7203.T",  # トヨタ自動車
    "9984.T",  # ソフトバンクグループ
    "6758.T"   # ソニーグループ
]
```

---

### 問題4: Phase 2で全パラメータが過学習フィルタで除外

**症状**: `filter_overfit_params()` で48組み合わせ全て除外

**対策**:
1. Phase 1に戻ってパラメータ空間を再検討
2. 成功基準緩和（非推奨）:
   - PF上限: 3.0 → 5.0
   - Win Rate: 40% → 30%
   - 取引数: 20 → 10

---

## 📚 次のステップ（Phase 1-3完了後）

### リアルトレード移行判断

**条件**:
- ✅ Phase 3で2パラメータ以上がOut-of-Sample通過
- ✅ 推奨パラメータのPF 2.0-3.0、Win Rate > 40%
- ✅ 10銘柄中8銘柄でPF > 1.0

**移行時期**: 2026年2月以降（Phase 3完了後1ヶ月間のペーパートレード推奨）

---

### エントリー戦略見直し（Phase 1-3失敗時）

**条件**:
- Phase 1で平均PF < 1.0
- Phase 2で全パラメータが過学習フィルタで除外
- Phase 3で全パラメータがOut-of-Sampleで劣化

**対策**:
- GC戦略 + トレンドフィルタ強化（Priority 2-A）
- BreakoutStrategy検証（Priority 2-B）
- 別エントリー戦略検討（MomentumStrategy、Mean Reversion等）

---

## 📝 関連ドキュメント

1. **EXIT_STRATEGY_REDESIGN_V2.md**: メインドキュメント（必読）
2. **PHASE6_PARAMETER_REFITTING_ANALYSIS.md**: Phase 3-6失敗分析
3. **EXIT_STRATEGY_VALIDATION_REPORT.md**: Phase 3検証レポート
4. **EXIT_STRATEGY_SEPARATION_DESIGN.md**: エントリー・エグジット分離設計

---

**作成者**: Backtest Project Team  
**バージョン**: 1.0  
**最終更新**: 2026年1月22日  
**ステータス**: クイックスタート完成
