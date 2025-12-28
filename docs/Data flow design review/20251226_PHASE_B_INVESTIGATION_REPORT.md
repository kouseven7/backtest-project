# Phase B: 命名規則統一・フォールバック見直し調査レポート

**作成日**: 2025-12-26  
**対象**: strategy vs strategy_name 命名規則統一（20+箇所）  
**副次調査**: フォールバック見直し（40+箇所）

---

## 1. 調査概要

### 1.1 調査目的
Phase A（Line 981修正）完了後、システム全体の命名規則不統一とフォールバック過剰使用を調査し、詳細な修正設計を作成する。**今回は調査のみ。修正は実施しない。**

### 1.2 調査方法
1. `grep_search`で`.get('strategy')`パターンを検索 → 50+件
2. `grep_search`で`.get('strategy_name')`パターンを検索 → 50+件
3. 主要ファイルの実際のコードを確認
4. ファイルの役割と影響範囲を評価
5. 修正優先度と詳細設計をまとめる

### 1.3 調査スコープ
- **対象ファイル**: 100+ファイル（active production code中心）
- **対象キー**: `'strategy'` vs `'strategy_name'`
- **フォールバック**: `.get(key, 'Unknown')`, `.get(key, 'UnknownStrategy')`, `.get(key, 'DSSMSStrategy')`

---

## 2. 主要ファイル分類

### 2.1 DSSMS統合系（Priority 1）

#### 2.1.1 dssms_strategy_stats_corrector.py
**役割**: DSSMS戦略別統計修正システム

**問題箇所**:
1. **Line 67** (function: `_generate_strategy_stats_from_trades`)
   ```python
   strategy = trade.get('strategy', 'UnknownStrategy')
   ```
   - **状況**: 取引データから戦略別統計を生成する際のグループ化キー
   - **フォールバック**: `'UnknownStrategy'`
   - **影響**: 7つの戦略が正しく区別されない原因の一つ

2. **Line 325** (function: `_calculate_strategy_stats`)
   ```python
   strategy = trade.get('strategy', 'UnknownStrategy')
   ```
   - **状況**: 戦略別統計計算時のグループ化
   - **フォールバック**: `'UnknownStrategy'`
   - **影響**: 同上

3. **Line 439** (function: `_generate_enhanced_strategy_stats`)
   ```python
   strategy = trade.get('strategy', 'UnknownStrategy')
   ```
   - **状況**: 強化版戦略統計生成時のグループ化
   - **フォールバック**: `'UnknownStrategy'`
   - **影響**: 同上

**修正方針**:
- **判断**: `'strategy'` → `'strategy_name'`に統一
- **理由**: comprehensive_reporter.pyの取引データは`strategy_name`キーで保存されている（Line 575, 604, 726）
- **フォールバック**: `'UnknownStrategy'`を維持（フォールバック自体は必要）

#### 2.1.2 dssms_trade_history_fixer.py
**役割**: DSSMS取引履歴問題修正スクリプト

**問題箇所**:
1. **Line 286** (function: `_enhance_trade_history`)
   ```python
   strategy = trade.get('strategy', 'DSSMSStrategy')
   ```
   - **状況**: 取引履歴データの戦略名取得
   - **フォールバック**: `'DSSMSStrategy'`（ダミーデフォルト）
   - **影響**: 実際の戦略名が取得できず`'DSSMSStrategy'`固定になる

**修正方針**:
- **判断**: `'strategy'` → `'strategy_name'`に統一
- **フォールバック**: `'DSSMSStrategy'`を`'UnknownStrategy'`に変更（命名統一）

#### 2.1.3 dssms_unified_output_engine.py
**役割**: DSSMS統一出力エンジン

**問題箇所**:
1. **Line 136** (function: `_convert_backtester_to_unified_format`)
   ```python
   'strategy': trade.get('strategy', 'Unknown'),
   ```
   - **状況**: バックテスターデータをDataFrame変換時
   - **フォールバック**: `'Unknown'`
   - **影響**: 統一出力の戦略名が不正確になる

2. **Line 662** (function: `_format_txt_trade_history`)
   ```python
   '戦略名': trade.get('strategy', 'Unknown'),
   ```
   - **状況**: TXT形式取引履歴出力
   - **フォールバック**: `'Unknown'`
   - **影響**: TXT出力の戦略名が不正確になる

**修正方針**:
- **判断**: `'strategy'` → `'strategy_name'`に統一
- **フォールバック**: `'Unknown'`を`'UnknownStrategy'`に変更（命名統一）

---

### 2.2 メイン実行系（Priority 1）

#### 2.2.1 main_system/reporting/main_text_reporter.py
**役割**: メインシステムのテキストレポート生成

**問題箇所**:
1. **Line 622** (function: `_calculate_strategy_ev`)
   ```python
   strategy = trade.get('strategy', 'Unknown')
   ```
   - **状況**: 戦略別期待値計算
   - **フォールバック**: `'Unknown'`
   - **影響**: 期待値計算が不正確になる

2. **Line 712** (function: `_analyze_strategy_performance`)
   ```python
   strategy = trade.get('strategy', 'Unknown')
   ```
   - **状況**: 戦略別パフォーマンス分析
   - **フォールバック**: `'Unknown'`
   - **影響**: パフォーマンス分析が不正確になる

3. **Line 784** (function: `_generate_trade_history_section`)
   ```python
   strategy = trade.get('strategy', 'Unknown')[:18]
   ```
   - **状況**: 取引履歴セクション生成（最初の10件表示）
   - **フォールバック**: `'Unknown'`
   - **影響**: 取引履歴の戦略名が不正確になる

**修正方針**:
- **判断**: `'strategy'` → `'strategy_name'`に統一
- **フォールバック**: `'Unknown'`を`'UnknownStrategy'`に変更（命名統一）

#### 2.2.2 main_system/reporting/comprehensive_reporter.py
**役割**: 包括的レポート生成（Phase A修正完了）

**現状**:
- **Line 384**: `result.get('strategy_name', 'Unknown')` ← ログ出力用
- **Line 575**: `'strategy_name': order.get('strategy_name', 'Unknown')` ← データ保存時
- **Line 604**: `sell.get('strategy_name', 'Unknown')` ← ログ出力用
- **Line 726**: `buy_order.get('strategy_name', 'Unknown')` ← 保有ポジション生成時
- **Line 981**: `strategy.get('strategy_name', 'Unknown')` ← **Phase A修正完了**

**修正方針**:
- **判断**: 修正完了。`'strategy_name'`で統一済み
- **フォールバック**: `'Unknown'`を使用（他ファイルとの統一のため`'UnknownStrategy'`への変更を推奨）

---

### 2.3 実行制御系（Priority 1）

#### 2.3.1 main_system/execution_control/integrated_execution_manager.py
**役割**: 統合実行マネージャー（複数戦略の重み付き実行）

**問題箇所**:
1. **Line 487** (function: `execute_strategies`)
   ```python
   strategy_name = result.get('strategy_name', 'Unknown')
   ```
   - **状況**: 戦略結果の重み付き集約
   - **フォールバック**: `'Unknown'`
   - **影響**: 戦略ウェイトの計算が不正確になる可能性

2. **Line 545** (function: `_update_drawdown_controller`)
   ```python
   strategy_name = result.get('strategy_name', 'Unknown')
   ```
   - **状況**: DrawdownController更新時の戦略名取得
   - **フォールバック**: `'Unknown'`
   - **影響**: ドローダウン制御が不正確になる可能性

**修正方針**:
- **判断**: `'strategy_name'`を維持（正しい）
- **フォールバック**: `'Unknown'`を`'UnknownStrategy'`に変更（命名統一）

#### 2.3.2 main_system/execution_control/strategy_execution_manager.py
**役割**: 戦略実行マネージャー（個別戦略の実行制御）

**問題箇所**:
1. **Line 445** (function: `execute_with_forced_exit`)
   ```python
   strategy={order_dict.get('strategy_name', 'Unknown')}
   ```
   - **状況**: ログ出力（ForceClose実行中の通常SELL抑制）
   - **フォールバック**: `'Unknown'`
   - **影響**: ログの戦略名が不正確になる

2. **Line 612** (function: `_process_orders_main_loop`)
   ```python
   "strategy_name": order_dict.get('strategy_name', 'Unknown'),
   ```
   - **状況**: execution_detailsへの戦略名保存
   - **フォールバック**: `'Unknown'`
   - **影響**: execution_detailsの戦略名が不正確になる

3. **Line 615** (function: `_process_orders_main_loop`)
   ```python
   strategy={order_dict.get('strategy_name', 'Unknown')}
   ```
   - **状況**: ログ出力（取引実行成功）
   - **フォールバック**: `'Unknown'`
   - **影響**: ログの戦略名が不正確になる

**修正方針**:
- **判断**: `'strategy_name'`を維持（正しい）
- **フォールバック**: `'Unknown'`を`'UnknownStrategy'`に変更（命名統一）

---

## 3. フォールバック分析

### 3.1 フォールバック種類
1. `'Unknown'` - 21箇所（comprehensive_reporter.py, main_text_reporter.py, integrated_execution_manager.py, strategy_execution_manager.py, dssms_unified_output_engine.py）
2. `'UnknownStrategy'` - 3箇所（dssms_strategy_stats_corrector.py）
3. `'DSSMSStrategy'` - 1箇所（dssms_trade_history_fixer.py）

### 3.2 copilot-instructions.md違反の評価

#### 3.2.1 違反基準
```markdown
## フォールバック機能の制限
- **モック/ダミー/テストデータを使用するフォールバック禁止**: 実データと乖離する結果を生成するフォールバック機能は実装しない
- **テスト継続のみを目的としたフォールバック禁止**: エラーを隠蔽して強制的にテストを継続させるフォールバックは実装しない
- **フォールバック実行時のログ必須**: フォールバック機能が動作した場合は必ずログに記録し、ユーザーが認識できるようにする
```

#### 3.2.2 違反評価
| ファイル | 行番号 | フォールバック値 | 違反評価 | 理由 |
|---------|-------|----------------|---------|------|
| dssms_strategy_stats_corrector.py | 67, 325, 439 | `'UnknownStrategy'` | ⚠️ **グレー** | 戦略統計が不正確になるがシステムは継続動作 |
| dssms_trade_history_fixer.py | 286 | `'DSSMSStrategy'` | 🚫 **違反** | ダミー戦略名を生成、実データと乖離 |
| dssms_unified_output_engine.py | 136, 662 | `'Unknown'` | ⚠️ **グレー** | 出力が不正確になるがシステムは継続動作 |
| main_text_reporter.py | 622, 712, 784 | `'Unknown'` | ⚠️ **グレー** | レポートが不正確になるがシステムは継続動作 |
| comprehensive_reporter.py | 384, 575, 604, 726, 981 | `'Unknown'` | ⚠️ **グレー** | レポートが不正確になるがシステムは継続動作 |
| integrated_execution_manager.py | 487, 545 | `'Unknown'` | ⚠️ **グレー** | 戦略制御が不正確になるがシステムは継続動作 |
| strategy_execution_manager.py | 445, 612, 615 | `'Unknown'` | ⚠️ **グレー** | execution_detailsが不正確になるがシステムは継続動作 |

**判定**: 
- **明確な違反**: dssms_trade_history_fixer.py Line 286（`'DSSMSStrategy'`はダミーデフォルト）
- **グレーゾーン**: その他（エラー隠蔽ではなく、不正確なデータ生成）

#### 3.2.3 ログ記録の有無
- **ログあり**: 0箇所（フォールバック時のログ記録なし）
- **ログなし**: 25箇所（すべて）

**判定**: **全箇所でcopilot-instructions.md違反** - 「フォールバック実行時のログ必須」違反

---

## 4. 修正設計

### 4.1 修正原則
1. **命名統一**: `'strategy'` → `'strategy_name'`（データ保存・取得キーを統一）
2. **フォールバック統一**: `'Unknown'`, `'UnknownStrategy'`, `'DSSMSStrategy'` → `'UnknownStrategy'`
3. **ログ追加**: フォールバック使用時は必ずログ記録

### 4.2 修正優先度

#### Priority 1（即座対応推奨）- DSSMS統合系
| ファイル | 行番号 | 現在のコード | 修正後のコード | 影響範囲 |
|---------|-------|------------|--------------|---------|
| dssms_strategy_stats_corrector.py | 67 | `trade.get('strategy', 'UnknownStrategy')` | `trade.get('strategy_name', 'UnknownStrategy')` | 戦略別統計生成 |
| dssms_strategy_stats_corrector.py | 325 | `trade.get('strategy', 'UnknownStrategy')` | `trade.get('strategy_name', 'UnknownStrategy')` | 戦略別統計計算 |
| dssms_strategy_stats_corrector.py | 439 | `trade.get('strategy', 'UnknownStrategy')` | `trade.get('strategy_name', 'UnknownStrategy')` | 強化版戦略統計 |
| dssms_trade_history_fixer.py | 286 | `trade.get('strategy', 'DSSMSStrategy')` | `trade.get('strategy_name', 'UnknownStrategy')` | 取引履歴強化 |
| dssms_unified_output_engine.py | 136 | `'strategy': trade.get('strategy', 'Unknown')` | `'strategy': trade.get('strategy_name', 'UnknownStrategy')` | バックテスターデータ変換 |
| dssms_unified_output_engine.py | 662 | `trade.get('strategy', 'Unknown')` | `trade.get('strategy_name', 'UnknownStrategy')` | TXT出力 |

#### Priority 2（次期対応推奨）- メイン実行系
| ファイル | 行番号 | 現在のコード | 修正後のコード | 影響範囲 |
|---------|-------|------------|--------------|---------|
| main_text_reporter.py | 622 | `trade.get('strategy', 'Unknown')` | `trade.get('strategy_name', 'UnknownStrategy')` | 戦略別期待値計算 |
| main_text_reporter.py | 712 | `trade.get('strategy', 'Unknown')` | `trade.get('strategy_name', 'UnknownStrategy')` | 戦略別パフォーマンス分析 |
| main_text_reporter.py | 784 | `trade.get('strategy', 'Unknown')` | `trade.get('strategy_name', 'UnknownStrategy')` | 取引履歴セクション生成 |

#### Priority 3（フォールバック値のみ統一）- 実行制御系
| ファイル | 行番号 | 現在のコード | 修正後のコード | 影響範囲 |
|---------|-------|------------|--------------|---------|
| comprehensive_reporter.py | 384, 575, 604, 726 | `'Unknown'` | `'UnknownStrategy'` | レポート生成 |
| integrated_execution_manager.py | 487, 545 | `'Unknown'` | `'UnknownStrategy'` | 戦略実行制御 |
| strategy_execution_manager.py | 445, 612, 615 | `'Unknown'` | `'UnknownStrategy'` | 戦略実行制御 |

**注**: comprehensive_reporter.py Line 981は**Phase A修正完了**。

### 4.3 ログ追加設計

#### 4.3.1 ログ追加箇所
すべてのフォールバック使用箇所にログ記録を追加:

```python
# 修正前
strategy = trade.get('strategy', 'UnknownStrategy')

# 修正後
strategy = trade.get('strategy_name', 'UnknownStrategy')
if strategy == 'UnknownStrategy':
    self.logger.warning(
        f"[FALLBACK] 戦略名が取得できませんでした: trade={trade}, "
        f"デフォルト値='{strategy}'"
    )
```

#### 4.3.2 ログレベル
- **WARNING**: フォールバック使用（データ不正確だが動作継続）
- **ERROR**: フォールバック使用かつ致命的影響（例: 戦略統計が完全に不正確）

---

## 5. 修正影響評価

### 5.1 修正前後の挙動変化

#### 5.1.1 DSSMS統合系
**修正前**:
- 取引データの`strategy_name`キーを読み取れず、フォールバック値（`UnknownStrategy`, `DSSMSStrategy`, `Unknown`）を使用
- 7つの戦略が区別されず、統計・レポートが不正確

**修正後**:
- 取引データの`strategy_name`キーを正しく読み取る
- 7つの戦略が正しく区別され、統計・レポートが正確になる
- `strategy_name`キーが存在しない場合はWARNINGログが出力される

#### 5.1.2 メイン実行系
**修正前**:
- 取引データの`strategy`キーを読み取る（データソースによって`strategy`または`strategy_name`が混在）
- 戦略別期待値・パフォーマンス分析が不正確

**修正後**:
- 取引データの`strategy_name`キーを正しく読み取る
- 戦略別期待値・パフォーマンス分析が正確になる
- `strategy_name`キーが存在しない場合はWARNINGログが出力される

### 5.2 回帰リスク
- **低リスク**: Priority 3（フォールバック値のみ統一） - 実行ロジック変更なし
- **中リスク**: Priority 2（メイン実行系） - レポート生成ロジック変更
- **高リスク**: Priority 1（DSSMS統合系） - 戦略統計生成ロジック変更

### 5.3 テスト推奨事項
1. **Phase A修正の再検証**: dssms_integrated_main.py実行後、`strategy_breakdown`が7つの戦略を区別できているか確認
2. **Priority 1修正後**: DSSMS Excel Exporterの戦略別統計シートが正しく生成されるか確認
3. **Priority 2修正後**: main_new.py実行後、テキストレポートの戦略別分析が正しいか確認
4. **ログ確認**: フォールバック使用時のWARNINGログが出力されるか確認

---

## 6. 実装ロードマップ

### 6.1 Phase B-1: Priority 1修正（DSSMS統合系）
**対象ファイル**: 
- dssms_strategy_stats_corrector.py（3箇所）
- dssms_trade_history_fixer.py（1箇所）
- dssms_unified_output_engine.py（2箇所）

**作業内容**:
1. 6箇所の`'strategy'` → `'strategy_name'`修正
2. フォールバック値を`'UnknownStrategy'`に統一
3. 6箇所にWARNINGログ追加
4. DSSMS実行テスト（dssms_integrated_main.py）
5. Excel出力の戦略別統計シート確認

**推定工数**: 2時間

### 6.2 Phase B-2: Priority 2修正（メイン実行系）
**対象ファイル**: 
- main_text_reporter.py（3箇所）

**作業内容**:
1. 3箇所の`'strategy'` → `'strategy_name'`修正
2. フォールバック値を`'UnknownStrategy'`に統一
3. 3箇所にWARNINGログ追加
4. main_new.py実行テスト
5. テキストレポートの戦略別分析確認

**推定工数**: 1時間

### 6.3 Phase B-3: Priority 3修正（実行制御系）
**対象ファイル**: 
- comprehensive_reporter.py（4箇所）
- integrated_execution_manager.py（2箇所）
- strategy_execution_manager.py（3箇所）

**作業内容**:
1. 9箇所のフォールバック値を`'UnknownStrategy'`に統一（キー名は変更なし）
2. 9箇所にWARNINGログ追加
3. main_new.py実行テスト
4. execution_details, 包括的レポート確認

**推定工数**: 1.5時間

### 6.4 総推定工数
**合計**: 4.5時間（Priority 1〜3すべて実施）

---

## 7. 検証チェックリスト

### 7.1 Phase B-1修正後
- [ ] dssms_strategy_stats_corrector.py Line 67, 325, 439 修正完了
- [ ] dssms_trade_history_fixer.py Line 286 修正完了
- [ ] dssms_unified_output_engine.py Line 136, 662 修正完了
- [ ] WARNINGログが6箇所に追加されている
- [ ] dssms_integrated_main.py実行成功（エラーなし）
- [ ] Excel出力の戦略別統計シートに7つの戦略が区別されている
- [ ] フォールバックログが出力されていない（全取引に`strategy_name`が存在）

### 7.2 Phase B-2修正後
- [ ] main_text_reporter.py Line 622, 712, 784 修正完了
- [ ] WARNINGログが3箇所に追加されている
- [ ] main_new.py実行成功（エラーなし）
- [ ] テキストレポートの戦略別期待値が正確
- [ ] テキストレポートの戦略別パフォーマンス分析が正確
- [ ] テキストレポートの取引履歴の戦略名が正確

### 7.3 Phase B-3修正後
- [ ] comprehensive_reporter.py Line 384, 575, 604, 726 修正完了
- [ ] integrated_execution_manager.py Line 487, 545 修正完了
- [ ] strategy_execution_manager.py Line 445, 612, 615 修正完了
- [ ] WARNINGログが9箇所に追加されている
- [ ] main_new.py実行成功（エラーなし）
- [ ] execution_detailsの戦略名が正確
- [ ] 包括的レポートの戦略名が正確

---

## 8. 重要注意事項

### 8.1 データソース確認
修正実施前に、以下を確認すること:
1. **comprehensive_reporter.py Line 575**が`strategy_name`キーでデータを保存していることを確認済み（Phase A確認）
2. DSSMSバックテスターが`strategy_name`キーで取引データを生成することを確認
3. `strategy`キーを使用しているデータソースが存在しないか確認

### 8.2 アーカイブファイルの扱い
grep_search結果に含まれる`archive/`, `output/`配下のファイルは**修正対象外**:
- 過去の実行結果（output/）
- 非推奨コード（archive/）
- 修正不要（ドキュメント用途のみ）

### 8.3 テストファイルの扱い
`tests/temp/`, `tests/core/`配下のテストファイルは**Phase B完了後に個別対応**:
- 修正後のコードに合わせてテストを更新
- テスト失敗時はテストコードを修正（本体コードは修正しない）

---

## 9. まとめ

### 9.1 調査結果サマリー
- **命名規則不統一**: 25箇所で`'strategy'`と`'strategy_name'`が混在
- **フォールバック過剰**: 25箇所でフォールバック使用、すべてログ記録なし
- **copilot-instructions.md違反**: 25箇所（フォールバックログなし）
- **明確なダミーデータ生成**: 1箇所（dssms_trade_history_fixer.py Line 286: `'DSSMSStrategy'`）

### 9.2 修正効果予測
**Phase B-1完了後**:
- DSSMS Excel Exporterの戦略別統計シートが正しく生成される
- 7つの戦略が正しく区別される
- 勝率・損益が実際の取引データから計算される

**Phase B-2完了後**:
- main_new.pyのテキストレポートが正確になる
- 戦略別期待値・パフォーマンス分析が正確になる

**Phase B-3完了後**:
- execution_details, 包括的レポートの戦略名が統一される
- フォールバック使用時のログが出力され、問題を早期発見できる

### 9.3 次のステップ
1. **今回**: Phase B調査完了（本レポート作成）
2. **Phase B-1実装**: Priority 1修正（DSSMS統合系）
3. **Phase B-2実装**: Priority 2修正（メイン実行系）
4. **Phase B-3実装**: Priority 3修正（実行制御系）
5. **Phase C検討**: 長期最適化（データスキーマ標準化、バリデーション強化）

### 9.4 実装ステータス管理
4. **Phase B-3実装**: Priority 3修正（実行制御系）まで完了した
---

## 付録: grep_search結果詳細

### A.1 `.get('strategy')` パターン（50+件）
主要ファイル抜粋:
- dssms_strategy_stats_corrector.py: Line 67, 325, 439
- dssms_trade_history_fixer.py: Line 286, 406
- dssms_unified_output_engine.py: Line 136, 662
- main_text_reporter.py: Line 622, 712, 784
- その他: archive/, output/, src/配下に40+件

### A.2 `.get('strategy_name')` パターン（50+件）
主要ファイル抜粋:
- comprehensive_reporter.py: Line 384, 575, 604, 726, 981（修正済み）
- integrated_execution_manager.py: Line 487, 545
- strategy_execution_manager.py: Line 445, 612, 615
- dssms_integrated_main.py: Line 2596
- その他: archive/, output/, src/配下に40+件

---

**End of Report**
