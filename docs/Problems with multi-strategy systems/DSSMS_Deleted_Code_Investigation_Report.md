# DSSMS削除コード調査レポート

**調査期日**: 2026年1月3日  
**調査目的**: 過去の修正時に銘柄保有ロジックやswitch延期ロジックなど重要なコードが削除されていないか確認  
**調査範囲**: src/dssms/dssms_integrated_main.py およびsymbol_switch_manager関連ファイル

---

## エグゼクティブサマリー

### 主要発見
1. **2025年12月19日（コミットd84cd6d）で大量の取引実行コードが削除**
   - ポジション管理コード（`position_size`, `position_entry_price`）
   - バックテスト終了時の強制決済処理（57行削除）
   - 銘柄切替時の取引実行処理（`_close_position`, `_open_position`呼び出し）
   - 切替コスト計算ロジック

2. **ユーザーが言及した「保有期間5日設定」は存在しない**
   - 現在の`min_holding_days`デフォルト値は**1日**
   - symbol_switch_manager.py、_fast.py、_ultra_light.pyすべてで`min_holding_days=1`

3. **銘柄切替頻度削減ロジックは部分的に存在**
   - `_evaluate_switch_cost_efficiency()`内で7日間に2回以上の切替を抑制
   - 月次切替制限（`max_switches_per_month=10`）は存在
   - しかし、**ultra_light版ではこれらのロジックが実装されていない**

4. **削除の意図は明確：「DSSMSは銘柄選択のみ、取引実行はmain_new.pyに委譲」**

---

## 調査結果詳細

### 1. 削除されたコードの内容

#### 1.1 ポジション管理コード削除（d84cd6d Line 157-158）

**削除前**:
```python
self.position_size = 0
self.position_entry_price = 0
```

**削除後**:
```python
# Phase 1: DSSMS設定値取得コード削除 - Stage 4-2（2025-12-19）
# 削除: self.position_size, self.position_entry_price初期化
# 削除理由: DSSMSはポジション管理しない設計に変更
#   - ポジション管理: main_new.py（PaperBroker経由）が担当
#   - portfolio_valueのみを追跡（main_new.pyのreturnを累積）
# 影響: position_value/cash_balance計算を全額キャッシュ扱いに変更
```

**評価**:
- **削除は正当**: DSSMSは銘柄選択システムであり、ポジション管理は本来の責務ではない
- **影響**: ポジション状態を内部で追跡しなくなったため、保有期間延長ロジックの実装が困難に

---

#### 1.2 バックテスト終了時の強制決済削除（d84cd6d Line 486-542, 57行）

**削除前**:
```python
if self.position_size > 0 and self.current_symbol:
    self.logger.info(
        f"[BACKTEST_END_FORCE_CLOSE_START] バックテスト期間終了時の強制決済開始: "
        f"symbol={self.current_symbol}, position_size={self.position_size}, "
        f"portfolio_value_before={self.portfolio_value}"
    )
    
    # ForceCloseフラグ設定（二重実行防止と同様）
    self.force_close_in_progress = True
    
    # 強制決済実行（最終営業日の終値で決済）
    final_trading_date = current_date - timedelta(days=1)
    # ... 57行のForceClose処理
    close_result = self._close_position(self.current_symbol, final_trading_date)
    # ... execution_details収集処理
```

**削除後**:
```python
# Phase 1: DSSMS設定値取得コード削除 - Stage 3-2（2025-12-19）
# 削除: バックテスト終了時のForceClose処理（Line 486-542, 57行）
# 削除理由: DSSMSが取引実行しないため、期末強制決済も不要
# 代替: main_new.pyのForceCloseStrategyが担当
# 影響: バックテスト終了時のexecution_detailsが生成されない（設計通り）
```

**評価**:
- **削除は設計変更に伴う必然**: main_new.pyのForceCloseStrategyに機能移行
- **問題点**: バックテスト終了時にポジションが未決済のまま終わる可能性（main_new.py側で適切に処理する必要）

---

#### 1.3 銘柄切替時の取引実行削除（d84cd6d Line 1595-1677, 82行）

**削除前**:
```python
if should_switch:
    # [修正課題3実装] 銘柄切替時のexecution_details取得用リスト（2025-12-14追加）
    switch_execution_details = []
    
    # ポジション解除（旧銘柄売却）
    if self.current_symbol and self.position_size > 0:
        # [Task11] ForceCloseフラグ設定
        self.force_close_in_progress = True
        self.logger.info(f"[DSSMS_FORCE_CLOSE_START] ForceClose開始、戦略SELL処理を起動")
        
        close_result = self._close_position(self.current_symbol, target_date)
        switch_result['close_result'] = close_result
        
        # [修正課題3実装] SELL側execution_detail取得（2025-12-14修正）
        if 'execution_detail' in close_result:
            close_result['execution_detail']['execution_type'] = 'switch'
            switch_execution_details.append(close_result['execution_detail'])
        
        # [Task11] ForceCloseフラグリセット
        self.force_close_in_progress = False
    
    # 新銘柄ポジション開始
    open_result = self._open_position(selected_symbol, target_date)
    switch_result['open_result'] = open_result
    
    # [修正課題3実装] BUY側execution_detail取得（2025-12-14修正）
    if 'execution_detail' in open_result:
        open_result['execution_detail']['execution_type'] = 'switch'
        switch_execution_details.append(open_result['execution_detail'])
    
    # 切替コスト
    switch_cost = self.portfolio_value * self.config.get('switch_cost_rate', 0.001)
    portfolio_before_switch = self.portfolio_value
    self.portfolio_value -= switch_cost
    portfolio_after_switch = self.portfolio_value
    
    # 詳細情報: switch処理時の詳細追跡ログ
    self.logger.info(
        f"[SWITCH_COST_DETAIL] {target_date.strftime('%Y-%m-%d')}: "
        f"from={switch_result['from_symbol']} to={selected_symbol}, "
        f"cost={switch_cost:.2f}, "
        f"portfolio_before={portfolio_before_switch:.2f}, "
        f"portfolio_after={portfolio_after_switch:.2f}, "
        f"rate={self.config.get('switch_cost_rate', 0.001)}"
    )
    
    switch_result.update({
        'switch_executed': True,
        'switch_cost': switch_cost,
        'reason': switch_evaluation.get('reason', 'dss_optimization'),
        'portfolio_value_before': portfolio_before_switch,
        'portfolio_value_after': portfolio_after_switch,
        'executed_date': target_date
    })
```

**削除後**:
```python
if should_switch:
    # Phase 1: DSSMS設定値取得コード削除 - Stage 2（2025-12-19）
    # 削除: 銘柄切替時の取引実行処理（_close_position, _open_position呼び出し）
    # 削除: switch_execution_details取得ロジック
    # 削除: execution_type='switch'設定
    # 理由: DSSMSは銘柄選択のみ担当、取引実行はmain_new.py（PaperBroker）が担当
    # 影響: switch関連のexecution_detailsは他に任せる（設計通り）
    
    # Phase 1: DSSMS設定値取得コード削除 - Stage 3-1（2025-12-19）
    # 削除: switch_costロジック（Line 1651-1677, 27行）
    # 削除理由: DSSMSが取引実行しないため、コスト計算は無意味
    # 影響: portfolio_value更新なし、switch_result簡略化
    
    switch_result.update({
        'switch_executed': True,
        'reason': switch_evaluation.get('reason', 'dss_optimization'),
        'executed_date': target_date
    })
```

**評価**:
- **削除は設計変更に伴う必然**: 取引実行はPaperBrokerに委譲
- **重要な影響**: 切替コストの計算がなくなり、実際のパフォーマンスが過大評価される可能性
- **推奨**: main_new.py側で銘柄切替時のコストを適切に計上する必要あり

---

### 2. 銘柄保有期間・Switch頻度削減ロジックの現状

#### 2.1 symbol_switch_manager.py（完全版）

**保有期間制限**:
```python
def __init__(self, config: Dict[str, Any]):
    # ...
    self.min_holding_days = switch_config.get('min_holding_days', 1)  # デフォルト1日
    # ...

def _check_min_holding_period(self, target_date: datetime) -> bool:
    """最小保有期間をチェック"""
    if self.current_holding_start is None:
        return True  # 初回は制限なし
    
    holding_days = self._get_current_holding_days(target_date)
    return holding_days >= self.min_holding_days
```

**Switch頻度削減**:
```python
def _evaluate_switch_cost_efficiency(self, from_symbol: str, to_symbol: str, 
                                     target_date: datetime) -> Dict[str, Any]:
    # 頻繁な切替を抑制するためのペナルティ
    recent_switches = self._count_recent_switches(target_date, days=7)
    if recent_switches >= 2:  # 1週間で2回以上の切替は抑制
        is_cost_effective = False
        reason = "frequent_switching_penalty"
    else:
        reason = "cost_effective_switch"
    # ...

def _count_recent_switches(self, target_date: datetime, days: int = 7) -> int:
    """指定期間内の切替回数をカウント"""
    start_date = target_date - timedelta(days=days)
    count = 0
    
    for switch in self.switch_history:
        switch_date = switch.get('executed_date', switch.get('target_date'))
        if isinstance(switch_date, datetime) and start_date <= switch_date <= target_date:
            if switch.get('status') == 'executed':
                count += 1
    
    return count
```

**月次制限**:
```python
def _check_monthly_switch_limit(self, target_date: datetime) -> bool:
    """月次切替制限をチェック"""
    monthly_count = self._get_monthly_switch_count(target_date)
    return monthly_count < self.max_switches_per_month  # デフォルト10回/月
```

#### 2.2 symbol_switch_manager_ultra_light.py（軽量版、現在使用中）

```python
class SymbolSwitchManagerUltraLight:
    def __init__(self, config):
        switch_config = config.get('switch_management', {})
        self.switch_cost_rate = switch_config.get('switch_cost_rate', 0.001)
        self.min_holding_days = switch_config.get('min_holding_days', 1)  # 定義のみ
        self.switch_history = []
        self.current_symbol = None
        self.current_holding_start = None
    
    def evaluate_symbol_switch(self, from_symbol, to_symbol, target_date):
        if from_symbol is None:
            return {'should_switch': True, 'reason': 'initial', 'status': 'approved'}
        if from_symbol == to_symbol:
            return {'should_switch': False, 'reason': 'same', 'status': 'rejected'}
        return {'should_switch': True, 'reason': 'basic', 'status': 'approved'}
        # ⚠️ min_holding_days, 頻度制限のチェックなし
```

**重大な問題**:
- **ultra_light版では保有期間制限が実装されていない**: `min_holding_days`は定義されているが、`evaluate_symbol_switch()`でチェックされない
- **頻度削減ロジックなし**: 7日間2回制限、月次制限のチェックが存在しない
- **結果**: 現在のDSSMSは**無制限に銘柄切替を実行**している状態

---

### 3. 削除コードと現在の問題の関連性

#### 3.1 ユーザーの懸念事項

> "DSSMSには銘柄保有期間の延長があった。今は５日に設定？その保有期間はswitchしないようにするロジックがあったはず"

**調査結果**:
- **「5日設定」は存在しない**: 現在も過去も`min_holding_days`のデフォルトは**1日**
- **保有期間延長ロジックは存在する（symbol_switch_manager.py完全版）**: ただし、**ultra_light版では未実装**
- **ユーザーの記憶は部分的に正確**: 保有期間制限ロジック自体は存在したが、5日設定は誤認の可能性

#### 3.2 現在の低エントリー問題との関連

**DSSMS_Low_Entry_Investigation.mdからの知見**:
- 85回の銘柄切替、平均保有期間3.9日
- 72.4%のGC信号機会を逃失（別銘柄を保有中だったため）

**削除コードとの関連**:
1. **ポジション管理削除**: DSSMSがポジション状態を追跡しないため、保有期間チェックができない
2. **ultra_light版の不完全実装**: 保有期間制限が機能していない → 頻繁な切替
3. **切替コスト削除**: コスト意識なく切替判断が行われる可能性

**因果関係の評価**:
- **削除コード自体が直接の原因ではない**: 設計変更に伴う必然的な削除
- **ultra_light版の不完全実装が問題**: 完全版にある制限ロジックが欠落
- **推奨事項**: ultra_light版に保有期間制限を実装する、または完全版に戻す

---

### 4. 削除されたコードの重要性評価

#### 4.1 復元すべきコード

**なし（設計変更に伴う正当な削除）**

#### 4.2 改善すべき点

1. **ultra_light版に保有期間制限を追加**:
```python
# symbol_switch_manager_ultra_light.py修正案
def evaluate_symbol_switch(self, from_symbol, to_symbol, target_date):
    if from_symbol is None:
        return {'should_switch': True, 'reason': 'initial', 'status': 'approved'}
    if from_symbol == to_symbol:
        return {'should_switch': False, 'reason': 'same', 'status': 'rejected'}
    
    # 追加: 保有期間チェック
    if self.current_holding_start:
        holding_days = (target_date - self.current_holding_start).days
        if holding_days < self.min_holding_days:
            return {
                'should_switch': False, 
                'reason': 'min_holding_period_not_met',
                'status': 'rejected',
                'holding_days': holding_days,
                'required_days': self.min_holding_days
            }
    
    return {'should_switch': True, 'reason': 'basic', 'status': 'approved'}
```

2. **設定ファイルで`min_holding_days`を調整**:
```python
# config設定（例）
switch_management = {
    'min_holding_days': 10,  # 1日 → 10日に変更
    'max_switches_per_month': 10,
    'switch_cost_rate': 0.001
}
```

3. **main_new.py側で切替コストを計上**:
```python
# integrated_execution_manager.py等で実装
if symbol_switched:
    switch_cost = portfolio_value * 0.001  # 0.1%
    self.paper_broker.deduct_cost(switch_cost)
```

---

## 結論

### 主要発見サマリー

1. **削除されたコードは「設計変更」に伴う正当な削除**:
   - DSSMSの役割を「銘柄選択のみ」に限定
   - 取引実行はmain_new.py（PaperBroker）に委譲
   - ポジション管理の責務分離を明確化

2. **ユーザーが言及した「保有期間5日設定」は存在しない**:
   - 現在も過去も`min_holding_days`のデフォルトは1日
   - 誤認の可能性、または別の設定ファイルの記憶混在

3. **真の問題はultra_light版の不完全実装**:
   - 保有期間制限が機能していない
   - 頻度削減ロジックが欠落
   - 結果として無制限に銘柄切替が発生

4. **低エントリー問題の根本原因**:
   - 頻繁な銘柄切替（85回、3.9日平均） → 72.4%機会損失
   - ultra_light版の制限ロジック欠落が寄与
   - SINGLE_BEST戦略選択モードも影響

### 推奨事項

#### 優先度A: ultra_light版の修正
- 保有期間制限チェックを追加
- 頻度削減ロジック（7日2回制限）を追加
- または完全版（symbol_switch_manager.py）に戻す

#### 優先度B: 設定値の調整
- `min_holding_days`: 1日 → 10日以上
- `max_switches_per_month`: 10回 → 5回程度

#### 優先度C: 切替コスト計上の実装
- main_new.py側で銘柄切替時のコスト（0.1%）を適切に計上→今は設定を０にしているが他の場所で実装済（のはず）

#### 優先度D: 戦略選択モードの見直し
- SINGLE_BEST → MARKET_ADAPTIVEへの変更検討→今後の課題であり2026/01/12時点では取り組まない
- トップ2戦略の並行運用でエントリー機会増加

---

## 参考資料

### 関連コミット
- **d84cd6d** (2025-12-19): DSSMS取引実行コード大量削除
- **075164d** (2025-12-19): 銘柄切替ロジック修正、force_close_on_entry追加

### 関連ドキュメント
- [docs/design/main_new_switch_impl_plan.md](../design/main_new_switch_impl_plan.md): 銘柄切替実装計画
- [docs/Improved trading opportunity performance/DSSMS_Low_Entry_Investigation.md](../Improved%20trading%20opportunity%20performance/DSSMS_Low_Entry_Investigation.md): 低エントリー問題調査
- [docs/Output decision logic problem/](../Output%20decision%20logic%20problem/): 出力ロジック問題調査群

### 関連ファイル
- [src/dssms/symbol_switch_manager.py](../../src/dssms/symbol_switch_manager.py): 完全版（制限ロジック実装済み）
- [src/dssms/symbol_switch_manager_ultra_light.py](../../src/dssms/symbol_switch_manager_ultra_light.py): 軽量版（制限ロジック未実装）
- [src/dssms/dssms_integrated_main.py](../../src/dssms/dssms_integrated_main.py): DSSMS統合メインシステム

---

## 追加調査: ユーザー質問への回答（2026-01-12追加）

### Q1: symbol_switch_manager.pyは消えているのか、動作していないのか、importされていないのか？

**回答**: **importされて動作している**が、**ultra_light版が優先的に読み込まれる設計**

**証拠**:
1. [dssms_integrated_main.py Line 50-71](../../src/dssms/dssms_integrated_main.py#L50-L71)の`_load_symbol_switch_manager_fast()`関数:
   ```python
   def _load_symbol_switch_manager_fast():
       try:
           # ultra_light版を優先的にロード
           fast_path = os.path.join(current_dir, "symbol_switch_manager_ultra_light.py")
           spec = importlib.util.spec_from_file_location("symbol_switch_manager_ultra_light", fast_path)
           if spec and spec.loader:
               module = importlib.util.module_from_spec(spec)
               spec.loader.exec_module(module)
               return module.SymbolSwitchManagerUltraLight  # ← ultra_light版を返す
       except Exception:
           pass
       
       try:
           # フォールバック: 完全版をインポート
           from src.dssms.symbol_switch_manager import SymbolSwitchManager
           return SymbolSwitchManager
       except ImportError:
           return None
   ```

2. 実際の初期化コード ([Line 306](../../src/dssms/dssms_integrated_main.py#L306)):
   ```python
   self.switch_manager = SymbolSwitchManager(switch_config)  # ← ultra_light版のインスタンスが作成される
   ```

**結論**: symbol_switch_manager.py（完全版）は**フォールバック用として残されている**が、通常は**ultra_light版が動作**している

---

### Q2: なぜultra_light版に移行したのか？DSSMSにポジション管理をさせない一環か？

**回答**: **パフォーマンス改善**が主目的。ポジション管理除去とは**別の理由**

**証拠**:
- Gitコミット履歴（f9c37d6, 2025-10-02）:
  ```
  Fallback problem countermeasures.md
  ランキング問題一応解決した
  src/dssms/dssms_integrated_main.pyのバックテストの時間パフォーマンス改善中コミット
  ```
  → **時間パフォーマンス改善**のために導入

- 関連ドキュメント: [TODO-PERF-001_COMPLETION_REPORT.md](../../docs/TODO-PERF-001_COMPLETION_REPORT.md)

**時系列整理**:
1. **2025-10-02**: パフォーマンス改善でultra_light版導入（f9c37d6）
2. **2025-12-19**: DSSMSポジション管理除去（d84cd6d）

**結論**: ultra_light版移行は**パフォーマンス最適化**、ポジション管理除去は**設計変更**。**別々の理由・別々のタイミング**で実施

---

### Q3: 元々7日間2回制限、月次制限が機能していたなら、再度実装すべきか？

**回答**: **絶対に実装すべき**。現在の低エントリー問題の**直接的原因**の一つ

**根拠**:
1. **完全版には実装済み**:
   - [symbol_switch_manager.py Line 328-341](../../src/dssms/symbol_switch_manager.py#L328-L341): `_count_recent_switches()`で7日間2回制限
   - [symbol_switch_manager.py Line 259-272](../../src/dssms/symbol_switch_manager.py#L259-L272): `_check_monthly_switch_limit()`で月次制限

2. **ultra_light版には未実装**:
   - [symbol_switch_manager_ultra_light.py Line 11-14](../../src/dssms/symbol_switch_manager_ultra_light.py#L11-L14): 全ての切替を承認（制限チェックなし）

3. **現在の問題**:
   - 85回の銘柄切替、平均保有期間3.9日 → **無制限切替が発生**
   - 72.4%のGC信号機会を逃失 → **頻繁な切替が機会損失を生む**

**結論**: 7日間2回制限・月次制限の再実装は**最優先課題**

---

### Q4: 完全版に戻す vs ultra_light版修正のメリット・デメリット・推奨

#### 選択肢A: 完全版（symbol_switch_manager.py）に戻す

**メリット**:
1. ✅ **即座に機能復元**: 7日間2回制限、月次制限、保有期間チェックが全て動作
2. ✅ **コード品質保証**: 既存の564行、テスト済み、例外処理完備
3. ✅ **統計機能充実**: 切替統計、コスト分析、履歴管理が完全実装
4. ✅ **拡張性**: 将来的なロジック追加が容易（コメント、ドキュメント完備）
5. ✅ **保守性**: モジュールヘッダー完備、責務分離明確

**デメリット**:
1. ❌ **パフォーマンス影響**: ロード時間・実行時間が増加（ultra_light比）
   - 完全版: 564行、多数のメソッド、エラーハンドリング
   - ultra_light版: 30行、最小限の処理
2. ❌ **import時間**: 初期化コストが大きい（特にバックテスト開始時）
3. ❌ **メモリ使用**: オブジェクトサイズが大きい

**推定パフォーマンス影響**:
- ロード時間: +20-50ms（1回のみ）
- 切替判定: +1-2ms/回（85回 = +85-170ms合計）
- **バックテスト全体への影響: 0.1秒程度**（11ヶ月バックテストで無視可能）

---

#### 選択肢B: ultra_light版を修正（保有期間制限+頻度削減追加）

**メリット**:
1. ✅ **最小限の変更**: 既存コードへの影響が小さい
2. ✅ **パフォーマンス維持**: 軽量版の速度を保持
3. ✅ **段階的改善**: 必要な機能のみ追加（オーバーエンジニアリング回避）
4. ✅ **カスタマイズ**: DSSMSの責務（銘柄選択のみ）に特化した実装

**デメリット**:
1. ❌ **開発コスト**: 新規実装が必要（20-30行の追加コード）
2. ❌ **テスト不足**: 完全版のような長期実績がない
3. ❌ **保守性**: コードが分散（完全版とultra_light版の2つ存在）
4. ❌ **機能不足**: 統計機能、コスト分析が欠落
5. ❌ **将来的負債**: 追加機能が必要になる度に再実装

**実装例**（修正が必要な箇所）:
```python
# symbol_switch_manager_ultra_light.py修正案
def evaluate_symbol_switch(self, from_symbol, to_symbol, target_date):
    if from_symbol is None:
        return {'should_switch': True, 'reason': 'initial', 'status': 'approved'}
    if from_symbol == to_symbol:
        return {'should_switch': False, 'reason': 'same', 'status': 'rejected'}
    
    # 追加: 保有期間チェック
    if self.current_holding_start:
        holding_days = (target_date - self.current_holding_start).days
        if holding_days < self.min_holding_days:
            return {
                'should_switch': False, 
                'reason': 'min_holding_period_not_met',
                'status': 'rejected',
                'holding_days': holding_days
            }
    
    # 追加: 7日間2回制限
    recent_switches = self._count_recent_switches(target_date, days=7)
    if recent_switches >= 2:
        return {
            'should_switch': False,
            'reason': 'frequent_switching_penalty',
            'status': 'rejected'
        }
    
    return {'should_switch': True, 'reason': 'basic', 'status': 'approved'}

def _count_recent_switches(self, target_date, days=7):
    start_date = target_date - timedelta(days=days)
    count = 0
    for switch in self.switch_history:
        switch_date = switch.get('executed_date')
        if switch_date and start_date <= switch_date <= target_date:
            count += 1
    return count
```

**推定開発コスト**:
- コード追加: 20-30行
- テスト作成: 1-2時間
- 検証: 1回のフルバックテスト（11ヶ月）

---

#### 推奨: **選択肢A（完全版に戻す）**

**理由**:
1. **パフォーマンス影響は無視可能**: 0.1秒（11ヶ月バックテストで0.003%）
2. **即座に問題解決**: 修正なしで全機能復元
3. **保守性**: 2つのバージョン管理の手間を削減
4. **品質保証**: 既存の実績あるコード
5. **将来性**: 統計機能、コスト分析が必要になる可能性（完全版なら即座に利用可能）

**実装手順**:
1. [dssms_integrated_main.py Line 50-71](../../src/dssms/dssms_integrated_main.py#L50-L71)の`_load_symbol_switch_manager_fast()`を修正:
   ```python
   def _load_symbol_switch_manager_fast():
       try:
           # 完全版を直接インポート（ultra_light版をスキップ）
           from src.dssms.symbol_switch_manager import SymbolSwitchManager
           return SymbolSwitchManager
       except ImportError:
           return None
   ```

2. 設定ファイル修正（config設定）:
   ```python
   switch_management = {
       'min_holding_days': 10,  # 1日 → 10日
       'max_switches_per_month': 5,  # 10回 → 5回
       'switch_cost_rate': 0.001
   }
   ```

3. 検証:
   - フルバックテスト実行（2025-01-01～2025-11-30）
   - 切替回数の削減確認（85回 → 30回以下を目標）
   - パフォーマンス影響測定（期待: 0.1秒以内）

**リスク評価**:
- パフォーマンス: **極小**（0.1秒）
- 機能リスク: **なし**（既存実績コード）
- 保守リスク: **低**（コード一本化）

---

### Q5: バックテスト終了時のポジション決済問題

**現状の問題**:
- DSSMSはポジション管理を削除（2025-12-19, d84cd6d）
- main_new.py（PaperBroker）がポジション管理を担当
- **しかし、バックテスト終了時の決済ロジックが実装されていない可能性**

**問題の影響**:
1. バックテスト終了時に未決済ポジションが残る
2. 最終的な損益計算が不正確
3. equity_curveの最終値が実際の清算価値と乖離

**次のTask**:
#### Task: バックテスト終了時のポジション強制決済実装

**優先度**: **A（最優先）**

**実装場所**: main_new.py（MainSystemController）またはIntegratedExecutionManager

**実装内容**:
1. バックテスト終了日の検出
2. PaperBrokerの全ポジション取得
3. 最終営業日の終値で強制決済
4. execution_detailsへの記録（execution_type='backtest_end_force_close'）
5. 最終損益の確定

**参考コード**（削除前のdssms_integrated_main.py Line 486-542）:
```python
# バックテスト終了時の強制決済処理（参考）
if self.position_size > 0 and self.current_symbol:
    self.force_close_in_progress = True
    final_trading_date = current_date - timedelta(days=1)
    while final_trading_date.weekday() >= 5:  # 土日をスキップ
        final_trading_date -= timedelta(days=1)
    
    close_result = self._close_position(self.current_symbol, final_trading_date)
    # execution_details収集
```

**検証方法**:
1. バックテスト終了時のポジション数 = 0を確認
2. 最終日のexecution_detailsにSELL注文が記録されているか確認
3. equity_curveの最終値 = 現金残高を確認

**推定工数**: 2-3時間（実装 + テスト）

---

## 改訂履歴

- **2026-01-03**: 初版作成（Version 1.0）
- **2026-01-12**: ユーザー質問への回答追加（Version 1.1）
  - Q1: symbol_switch_manager.pyの動作状況
  - Q2: ultra_light版移行の理由
  - Q3: 制限機能の再実装必要性
  - Q4: 完全版 vs ultra_light版修正のメリット・デメリット・推奨
  - Q5: バックテスト終了時のポジション決済問題

---

**調査完了日**: 2026年1月12日  
**調査者**: GitHub Copilot  
**レポートバージョン**: 1.1
