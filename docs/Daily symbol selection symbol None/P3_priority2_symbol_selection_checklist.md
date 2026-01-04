# P3調査: Priority 2 銘柄選択・戦略実行確認チェックリスト

## 📋 **調査対象**
**核心問題**: なぜ同じ`_get_optimal_symbol()`メソッドが：
- P2-2の個別実行では正常動作（1662、6954選択成功）するのに
- P3の統合実行ではNoneを返すのか

## 🎯 **優先度順チェックリスト**

### **【Priority 2-1】_get_optimal_symbol()統合実行時の内部動作詳細**
- [ ] **P2-1-A**: 統合実行時の`_get_optimal_symbol()`実際の戻り値確認
- [ ] **P2-1-B**: `_get_optimal_symbol()`内部の各ステップ実行状態確認
- [ ] **P2-1-C**: `ensure_dss_core()`呼び出し結果確認
- [ ] **P2-1-D**: `ensure_advanced_ranking()`呼び出し結果確認
- [ ] **P2-1-E**: 例外処理（Line 1614, 1618-1619）の実行状態確認

### **【Priority 2-2】target_date, target_symbolsパラメータの実際の値**
- [ ] **P2-2-A**: `target_date`の実際の値確認（2025-01-15）
- [ ] **P2-2-B**: `target_symbols`の実際の値確認（None or リスト）
- [ ] **P2-2-C**: P2-2個別実行時との差異確認
- [ ] **P2-2-D**: 日付フォーマット・型の違い確認

### **【Priority 2-3】DSS Core V3初期化状態の統合実行時の確認**
- [ ] **P2-3-A**: `self.dss_core`の実際の値確認（None確認済み）
- [ ] **P2-3-B**: `dss_available`グローバル変数の値確認
- [ ] **P2-3-C**: `_initialize_dss_core()`実行時の詳細動作確認
- [ ] **P2-3-D**: DSS Core V3インポート・インスタンス化の失敗原因特定

### **【Priority 2-4】execution_detailsが0件である根本原因**
- [ ] **P2-4-A**: `daily_result['execution_details']`の生成箇所確認
- [ ] **P2-4-B**: `_convert_main_new_result()`の呼び出し状態確認
- [ ] **P2-4-C**: マルチ戦略実行部分の到達状態確認
- [ ] **P2-4-D**: `symbol=None`による早期リターンの影響確認

## 🔧 **調査方法**

### **デバッグスクリプト作成パターン**
```python
# 統合実行時の_get_optimal_symbol()詳細ログ取得
def debug_get_optimal_symbol_integration():
    # 1. パラメータ確認
    # 2. 内部ステップ詳細ログ
    # 3. 例外処理確認
    # 4. 戻り値詳細確認
```

### **比較調査パターン**
```python
# P2-2個別実行 vs P3統合実行の詳細比較
def compare_execution_contexts():
    # 1. 環境状態比較
    # 2. パラメータ比較
    # 3. 初期化状態比較
    # 4. 実行フロー比較
```

## 📊 **証拠収集フォーマット**

各チェック項目は以下のフォーマットで報告:

```
【項目】: P2-1-A _get_optimal_symbol()統合実行時の実際の戻り値確認
【調査結果】: 〇〇を確認しました
【根拠】: ファイル名 Line XXX の実際の内容「△△△」
【P2-2との差異】: 個別実行時は「XXX」だが統合実行時は「YYY」
【判定】: ✅正常 / ❌異常 / ⚠️部分的 / ❓不明
```

## 🎯 **重要確認ポイント**

### **コード実行パス追跡**
1. **Line 712**: `selected_symbol = self._get_optimal_symbol(target_date, target_symbols)`
2. **Line 1555-1620**: `_get_optimal_symbol()`内部動作
3. **Line 1567**: `self.ensure_dss_core()`
4. **Line 1614, 1618-1619**: 例外処理箇所

### **P2-2成功パターンとの比較**
- **P2-2成功例**: 1662選択、6954選択の実行環境
- **P3失敗例**: None返却の実行環境
- **差異分析**: 何が違うのか？

## 🚨 **予想される調査結果**

### **仮説1**: DSS Core V3初期化失敗
- `self.dss_core = None`のため、DSS Core V3が使用できない
- フォールバック処理も失敗している

### **仮説2**: パラメータ差異
- `target_symbols`の違い（None vs 具体的リスト）
- `target_date`の型・フォーマットの違い

### **仮説3**: 環境・初期化順序の違い**
- 統合実行時と個別実行時の初期化順序が異なる
- 依存関係の初期化タイミングが異なる

---

**Status**: ✅ Priority 2チェックリスト作成完了  
**Next**: P2-1-A からの順次調査開始