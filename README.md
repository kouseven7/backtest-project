# My Backtest Project

## 必要なライブラリ
- pandas
- openpyxl
- yfinance

## 実行方法
1. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```

## GitHub Copilot Workspace について
本プロジェクトでは、GitHub Copilot Workspace（GitHub Copilot HQ）を活用した開発を推奨しています。
詳細については、[GitHub Copilot Workspace ドキュメント](docs/GitHubCopilotWorkspace.md)を参照してください。

## [FIRE] プロジェクト最終目的（再定義）
本プロジェクトの最終目的は、DSSMS（Dynamic Stock Selection Multi-Strategy System）を用いて
日経225構成銘柄から日次で最適上昇トレンド銘柄を動的選択し、マルチ戦略スタックを適用、
kabu STATION API を通じて継続的なリスク調整後利益を獲得することです。

### システム階層
1. DSSMS Core (Screening / Ranking / Scoring / Switching)
2. Multi-Strategy Execution Layer（既存戦略適用）
3. Realtime & kabu Integration
4. Analytics & Reporting (Excel / 評価 / KPI)

### 日次オペレーション（概念フロー）
1. スクリーニング（~225→50銘柄）
2. パーフェクト/準パーフェクトオーダー階層分類
3. 総合スコアリング + 適応補正
4. 最適銘柄 + バックアップ5抽出
5. マルチ戦略適用（保持/切替判定）
6. パフォーマンス & リスク計測
7. ログ / Excel / KPI更新

### 開発進行ステータス（2025-現在）
- 設計文書: 充足
- コアクラス: 60%（未実装/空メソッドあり）
- 出力/統計式: 要修復（数式分母欠落 等）
- ライブ統合: 認証/骨格 8割、発注安全化ロジック未固化

### 既知の重要課題（優先順）
1. 出力エンジン重複 & 不整合（統計数式未完備）  
2. 切替履歴計測の信頼性確保（保有時間計算統一）  
3. 数学的指標（勝率/Profit Factor/平均損益）再実装  
4. Realtime Execution / kabu manager の空メソッド充足  
5. Perfect Order 検出検証用ミニテストスイート追加

## 🔁 Purpose Hierarchy（目的階層）
1. Ultimate Business Goal: 日次DSSMS選択 + マルチ戦略実行 + kabu発注でリスク調整後利益最大化  
2. System Goal (DSSMS Core): 50銘柄ランキング→最適銘柄+バックアップ抽出→切替判定→ログ/学習  
3. Technical Objectives:
   - ランキング完了 < 30秒
   - 日次フロー成功率 > 99%
   - 再現性（決定論モード）保証
   - 切替理由メタデータ出力

## [CHART] KPI & 定義
| カテゴリ | 指標 | 初期ターゲット | 定義/補足 |
|----------|------|----------------|-----------|
| 正確性 | Perfect Order 判定一致率 | 99% | 手動検証20銘柄との一致 |
| 効率 | 50銘柄ランキング時間 | <30s | キャッシュ未使用基準 |
| 切替品質 | 不要切替率 | <20% | 不要切替=切替後10営業日累積利益率 ≤ 想定取引コスト(往復) |
| リスク | 最大DD改善率 | >20% | 固定銘柄比較 |
| 運用 | 日次成功率 | >99% | 致命エラーなく全ステップ完了 |
| 出力 | Excel再計算一致率 | 100% | 内部再計算 vs 出力値 |

不要切替の閾値: (利益率 - 取引コスト) ≤ 0 を「不要」。  
取引コストは暫定 0.2% 往復（調整可能）。

## [TEST] 切替評価フレーム（概念）
1. 切替イベント保存 (旧/新銘柄, 理由, スコア差, 期待α)  
2. 10営業日後差分検証 (実績α - コスト)  
3. 不要切替率集計 / 月次報告  

## ⚠ エラーハンドリング分類
| レベル | 基準 | アクション |
|--------|------|-----------|
| CRITICAL | データ欠落/整合性崩壊/発注異常 | 即停止・通知 |
| ERROR | 個別指標計算失敗（代替可能） | フォールバック・継続 |
| WARNING | 欠損補完/キャッシュ失効 | ログ記録 |
| INFO | 状態遷移 | ログのみ |
| DEBUG | 詳細解析 | 決定論モード時有効化 |

## 🛡 方針
- 分散投資なし / DSSMS単一最適銘柄集中運用（将来拡張余地あり）
- 強化学習導入「現時点なし」(設計余白のみ確保)
- kabu 実行タイミング固定せず（寄前/引後/イベントトリガ両対応）

## 🧪 テスト管理

### テストフォルダ構造
```
tests/
├── core/          # 継続的なテスト（回帰テスト、CI/CD）
├── integration/   # 統合テスト（将来使用）
└── temp/          # 一時テスト（成功後削除）
```

### 一時テスト (tests/temp/)
- **用途**: 新機能の動作確認、一度のみの検証
- **命名**: `test_YYYYMMDD_<feature>.py`
- **削除方法**: `python tests/cleanup_temp_tests.py`
- **詳細**: [一時テスト管理ガイド](docs/TEMP_TEST_MANAGEMENT.md)

### 継続テスト (tests/core/)
- **用途**: 回帰テスト、CI/CD自動テスト
- **実行**: `pytest tests/core/` または個別実行

### テスト作成ガイドライン
詳細は `.github/copilot-instructions.md` の「テストファイル配置ルール」を参照

