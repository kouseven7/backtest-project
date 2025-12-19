"""
main_new.py側の実装計画

Phase 2: 銘柄切替処理の実装
============================

【新規実装】

1. 銘柄切替通知の受信
   - DSSMSから選択銘柄を受け取る
   - 既存: execute_comprehensive_backtest(ticker, ...)
   - 拡張: 銘柄切替フラグの追加

2. ForceClose戦略の拡張
   - 現状: バックテスト終了時のみForceClose
   - 追加: 銘柄切替時のForceClose
   - トリガー: DSSMSからの切替通知

3. 新銘柄のエントリー判断
   - 各戦略が独自にエントリー判断
   - DSSMSは銘柄を渡すのみ
   - 強制エントリーはしない

【実装案】

Option 1: execute_comprehensive_backtest()拡張
```python
def execute_comprehensive_backtest(
    self,
    ticker: str,
    stock_data: Optional[pd.DataFrame] = None,
    index_data: Optional[pd.DataFrame] = None,
    days_back: int = 365,
    backtest_start_date: Optional[datetime] = None,
    backtest_end_date: Optional[datetime] = None,
    warmup_days: int = 90,
    force_close_on_entry: bool = False  # 新規追加
) -> Dict[str, Any]:
    '''
    force_close_on_entry: True時は既存ポジションを強制決済してから開始
    '''
    if force_close_on_entry:
        # PaperBrokerの全ポジション決済
        self._force_close_all_positions()
```

Option 2: 新メソッド switch_symbol()追加
```python
def switch_symbol(
    self,
    new_ticker: str,
    switch_date: datetime,
    stock_data: Optional[pd.DataFrame] = None,
    index_data: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    '''
    銘柄切替専用メソッド
    1. 既存ポジション決済
    2. 新銘柄のバックテスト開始
    '''
    # 1. 既存ポジション決済
    close_results = self._force_close_all_positions()
    
    # 2. 新銘柄のバックテスト開始
    backtest_results = self.execute_comprehensive_backtest(
        ticker=new_ticker,
        stock_data=stock_data,
        index_data=index_data,
        backtest_start_date=switch_date,
        backtest_end_date=switch_date  # 1日のみ
    )
    
    return {
        'close_results': close_results,
        'backtest_results': backtest_results
    }
```

【PaperBroker拡張】

1. 全ポジション決済メソッド追加
```python
# paper_broker.py
def close_all_positions(self, current_date: datetime) -> List[Dict[str, Any]]:
    '''
    全ポジションを決済
    
    Returns:
        List[Dict]: 決済結果のリスト
    '''
    results = []
    for symbol in list(self.positions.keys()):
        position = self.positions[symbol]
        
        # SELL注文作成
        sell_order = Order(
            id=str(uuid.uuid4()),
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=position['quantity'],
            status=OrderStatus.PENDING,
            created_at=current_date
        )
        
        # 注文実行
        success = self.submit_order(sell_order)
        
        if success:
            results.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'exit_price': self.get_current_price(symbol),
                'pnl': (self.get_current_price(symbol) - position['entry_price']) * position['quantity'],
                'order_id': sell_order.id
            })
    
    return results
```

【DSSMSとmain_new.pyの連携】

修正前:
```python
# dssms_integrated_main.py
if should_switch:
    close_result = self._close_position(self.current_symbol, target_date)  # DSSMSが実行
    open_result = self._open_position(selected_symbol, target_date)  # DSSMSが実行
```

修正後:
```python
# dssms_integrated_main.py
if should_switch:
    # 銘柄切替通知のみ（取引は実行しない）
    self.logger.info(f"[SYMBOL_SWITCH] {self.current_symbol} → {selected_symbol}")
    switch_result['switch_requested'] = True
    switch_result['from_symbol'] = self.current_symbol
    switch_result['to_symbol'] = selected_symbol
    switch_result['switch_date'] = target_date
    
    # main_new.pyに切替通知
    # NOTE: 既存のexecute_comprehensive_backtest()経由で新銘柄を渡すため、
    #       ここでは銘柄変更の記録のみ

# main_new.py側
# execute_comprehensive_backtest()で銘柄が変わったことを検出したら
# IntegratedExecutionManagerがPaperBrokerに既存ポジション決済を指示
```

【実装の優先順位】

Priority 1: PaperBrokerのclose_all_positions()実装
  - 全ポジション決済機能
  - 影響範囲: PaperBrokerのみ
  - リスク: 低

Priority 2: main_new.pyのswitch_symbol()実装
  - 銘柄切替専用メソッド
  - 影響範囲: MainSystemControllerのみ
  - リスク: 中

Priority 3: DSSMSの銘柄切替ロジック修正
  - 取引実行コード削除
  - switch_symbol()呼び出し追加
  - 影響範囲: 広範
  - リスク: 高

Priority 4: execution_type='switch'削除
  - execution_detailsの整理
  - レポート生成への影響確認
  - 影響範囲: 中
  - リスク: 中
"""
