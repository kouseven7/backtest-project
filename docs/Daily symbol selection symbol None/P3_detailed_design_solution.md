# P3解決方針詳細設計

**設計日時**: 2026-01-04  
**根本原因**: `daily_result['symbol']`初期化後の更新処理欠如  
**目標**: P3出力ファイル生成の成功

---

## 🎯 **設計方針の選択**

### **Option A: Switch処理後にdaily_result['symbol']を更新** [**推奨**]
```python
# Line 718以降のswitch処理後に追加
if switch_result.get('switch_executed', False):
    daily_result['switch_executed'] = True
    daily_result['symbol'] = self.current_symbol  # 追加: 正しい銘柄を設定
    self.switch_history.append(switch_result)
```

**メリット**:
- 最小限の変更で済む
- switch成功後に確実に正しい銘柄が設定される
- 既存ロジックとの整合性が高い

**デメリット**:
- switch処理が失敗した場合はNoneのまま

---

### **Option B: selected_symbolを直接設定** [**次点**]
```python
# Line 712-716の銘柄選択成功後に追加
selected_symbol = self._get_optimal_symbol(target_date, target_symbols)

if not selected_symbol:
    daily_result['errors'].append('銘柄選択失敗')
    return daily_result

daily_result['symbol'] = selected_symbol  # 追加: 選択された銘柄を設定
```

**メリット**:
- 銘柄選択成功時に即座に設定される
- switch処理の成否に依存しない

**デメリット**:
- switch処理でcurrent_symbolが変わる可能性との整合性問題

---

### **Option C: 初期化の遅延実行** [**非推奨**]
```python
# daily_result初期化をswitch処理後に移動
```

**メリット**:
- 論理的に正しいタイミングで初期化

**デメリット**:
- 大幅な構造変更が必要
- early return処理との整合性問題

---

## 📋 **推奨設計：Option A詳細仕様**

### **修正対象ファイル**
`src/dssms/dssms_integrated_main.py`

### **修正箇所**
Line 720付近（switch_result処理部分）

### **修正前コード**
```python
if switch_result.get('switch_executed', False):
    daily_result['switch_executed'] = True
    self.switch_history.append(switch_result)
```

### **修正後コード**
```python
if switch_result.get('switch_executed', False):
    daily_result['switch_executed'] = True
    # [P3修正] 銘柄切替成功後にdaily_result['symbol']を更新
    daily_result['symbol'] = self.current_symbol
    self.switch_history.append(switch_result)
```

### **修正理由**
- Line 688の初期化時はself.current_symbol=Noneのため、daily_result['symbol']=None
- switch処理でself.current_symbolは正しく'1662'に設定される
- daily_result['symbol']を更新しないと、P3出力でsymbol=Noneのまま処理される

### **影響範囲**
- `_process_daily_trading()`メソッドのみ
- 戻り値`daily_result['symbol']`の値が正確になる
- P3出力処理で正しいsymbol値が使用される

### **テストケース**
1. **正常ケース**: switch処理成功 → daily_result['symbol']='1662'
2. **異常ケース**: switch処理失敗 → daily_result['symbol']=None（変更なし）
3. **初期銘柄なしケース**: current_symbol=None → daily_result['symbol']=None（現状維持）

---

## 🔍 **設計検証**

### **副作用確認**
- **他メソッドへの影響**: なし（daily_result内の値のみ変更）
- **既存ロジックとの整合**: 既存のswitch処理ロジックと整合
- **エラーハンドリング**: switch失敗時は従来通りNoneのまま

### **パフォーマンス影響**
- **実行時間**: 無視できるレベル（変数代入のみ）
- **メモリ使用量**: 変化なし

### **運用上の注意点**
- switch処理が正常に動作することが前提
- self.current_symbolの値が正しいことが前提

---

## ✅ **実装準備**

### **実装手順**
1. 該当行の特定（Line 720付近）
2. コード修正の実行
3. デバッグテストの実行
4. P3統合テストの実行

### **検証方法**
- P3矛盾解析スクリプトでの再テスト
- 統合バックテスト実行でのP3出力確認

### **成功基準**
- daily_result['symbol']に正しい銘柄コード（'1662'等）が設定される
- P3出力フォルダにファイルが生成される
- symbol=Noneでのフィルタリング失敗が解消される

---

**Status**: ✅ **設計完了**  
**Next Action**: **実装実行**