# main_v2.py開発指示 - AIアシスタント用

## 🎯 **コア原則**
1. **クリーンスタート**: main.pyの問題を継承しない独立設計
2. **段階的開発**: Phase 1→2→3で確実に構築  
3. **実証必須**: 各段階で必ず`strategy.backtest()`実行確認
4. **モジュール再利用**: 調査済みモジュールを積極活用

## 📁 **プロジェクト構造**
```

├── main_v2.py          # メインエントリーポイント
├── config/             # 設定管理
├── strategies/         # 戦略実装  
├── output/             # 結果出力
└── data/               # データ処理
```

## 🚀 **Phase 1: 基礎実装**
- **対象**: VWAPBreakoutStrategy 単体のみ
- **必須**: Entry_Signal/Exit_Signal列生成確認
- **出力**: CSV+JSON+TXT形式（Excel禁止）
- **検証**: 実際のトレード回数 > 0確認

## ⚡ **Phase 2: 機能拡張**
- **戦略追加**: main.py実証済み7戦略を順次追加
- **参照**: `docs/Plan to create a new main entry point/main_py_modules_investigation.md`
- **確認**: 各戦略追加後に実バックテスト実行

## 🔧 **Phase 3: 完成**
- **モジュール活用**: `docs/Plan to create a new main entry point/reusable_modules_investigation_no_dssms.md`参照
- **高優先度**: logger_config, risk_management等18個
- **中優先度**: indicators, strategies等34個

## 🔄 **モジュール再利用戦略**
### **参照ドキュメント**
- **実証済み**: `docs/Plan to create a new main entry point/main_py_modules_investigation.md`（高優先度20個）
- **未使用候補**: `docs/Plan to create a new main entry point/reusable_modules_investigation_no_dssms.md`（高優先度18個）

### **再利用フロー（必須）**
1. **モジュール選定** → **テスト実行** → **結果判定**
2. **実用可** → **実装採用**
3. **不可+軽度修正可** → **修正** → **再テスト** → **実用可** → **実装採用**
4. **不可+修正困難** → **破棄**

### **再利用メリット**
- ✅ main.py複雑性回避
- ✅ main_v2.py実装工数削減  
- ✅ テスト確実性向上

## 🚨 **必須チェック項目**
- ✅ `strategy.backtest()`の実行確認
- ✅ Entry_Signal/Exit_Signal列の存在確認
- ✅ 実際のトレード件数 > 0の確認
- ✅ profit=0の場合は原因調査必須
- ✅ Unicodeエラー対策（Windows対応）

## ⚠️ **禁止事項**
- ❌ モックデータ・テストデータの残存
- ❌ DSSMS関連モジュールの使用
- ❌ Excel出力の実装
- ❌ 未検証でのコード追加

## 🎪 **開発手順**
1. **専用フォルダ作成**: `main_v2_project/`
2. **基礎構造実装**: main_v2.py + 最小構成
3. **VWAPBreakout単体テスト**: 実バックテスト確認
4. **段階的戦略追加**: 1戦略ずつ追加・検証
5. **未使用モジュール統合**: 調査レポート基準

---
**重要**: 各Phase完了時は必ず実際のバックテスト実行で動作確認すること

**注意**: このプロジェクトは実際のバックテスト実行を目的としています。実行を妨げる変更は避けてください。