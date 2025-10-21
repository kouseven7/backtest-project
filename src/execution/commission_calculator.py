"""
Commission Calculator - 手数料計算モジュール

三菱UFJ eスマート証券の手数料体系に基づく正確な手数料計算を提供します。
日本株の単元株制度（100株単位）にも対応しています。

主な機能:
- 約定代金に応じた段階的手数料計算
- 100万円超の場合の比例手数料計算（上限4,059円）
- 日本株の単元株制度対応（100株単位への調整）
- スリッページ計算（オプション）
- 取引コスト総額計算（手数料+スリッページ）

統合コンポーネント:
- TradeExecutor: 注文実行時の手数料計算
- PortfolioTracker: 取引履歴への手数料記録
- PaperBroker: ペーパートレードでの手数料シミュレーション
- StrategyExecutionManager: ポジションサイズ計算時の手数料考慮

セーフティ機能/注意事項:
- 負の約定代金は0円として扱う
- 単元未満株は100株に切り上げ（端数切り捨ても選択可能）
- 手数料は常に税込価格
- 100万円超の手数料は必ず上限4,059円を適用

Author: Backtest Project Team
Created: 2025-10-20
Last Modified: 2025-10-20
"""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def calculate_japanese_stock_commission(contract_value: float) -> float:
    """
    三菱UFJ eスマート証券の手数料計算（税込）
    
    約定代金に応じた段階的手数料体系:
    - 0円超～5万円以下: 55円
    - 5万円超～10万円以下: 99円
    - 10万円超～20万円以下: 115円
    - 20万円超～50万円以下: 275円
    - 50万円超～100万円以下: 535円
    - 100万円超: 約定金額×0.099% + 99円（上限4,059円）
    
    Args:
        contract_value (float): 約定代金（株数 × 株価）
        
    Returns:
        float: 手数料（税込、円）
        
    Example:
        >>> calculate_japanese_stock_commission(45000)  # 4.5万円
        55.0
        >>> calculate_japanese_stock_commission(900000)  # 90万円
        535.0
        >>> calculate_japanese_stock_commission(5000000)  # 500万円
        4059.0
    """
    if contract_value <= 0:
        return 0.0
    
    if contract_value <= 50000:
        return 55.0
    elif contract_value <= 100000:
        return 99.0
    elif contract_value <= 200000:
        return 115.0
    elif contract_value <= 500000:
        return 275.0
    elif contract_value <= 1000000:
        return 535.0
    else:
        # 100万円超: 約定金額×0.099% + 99円（上限4,059円）
        commission = contract_value * 0.00099 + 99.0
        return min(commission, 4059.0)


def adjust_to_trading_unit(quantity: float, unit_size: int = 100, 
                           round_up: bool = True) -> int:
    """
    日本株の単元株制度に合わせて株数を調整
    
    日本株は通常100株が1単元（最低取引単位）となっています。
    この関数は任意の株数を単元株数に調整します。
    
    Args:
        quantity (float): 希望株数
        unit_size (int): 1単元の株数（デフォルト100株）
        round_up (bool): True=切り上げ、False=切り捨て
        
    Returns:
        int: 単元株数に調整された株数
        
    Example:
        >>> adjust_to_trading_unit(150, round_up=True)  # 切り上げ
        200
        >>> adjust_to_trading_unit(150, round_up=False)  # 切り捨て
        100
        >>> adjust_to_trading_unit(250)
        300
    """
    if quantity <= 0:
        return 0
    
    if round_up:
        # 切り上げ（単元未満を次の単元へ）
        units = int((quantity + unit_size - 1) / unit_size)
    else:
        # 切り捨て（単元未満を切り捨て）
        units = int(quantity / unit_size)
    
    return units * unit_size


def calculate_max_affordable_quantity(
    available_funds: float,
    stock_price: float,
    unit_size: int = 100,
    include_slippage: bool = True,
    slippage_rate: float = 0.0001  # 0.01%
) -> Tuple[int, float, float, float]:
    """
    利用可能資金から購入可能な最大株数を計算（手数料・スリッページ込み）
    
    二分探索を使用して、手数料とスリッページを考慮した上で
    利用可能資金内で購入できる最大の株数を計算します。
    
    Args:
        available_funds (float): 利用可能資金（円）
        stock_price (float): 株価（円）
        unit_size (int): 1単元の株数（デフォルト100株）
        include_slippage (bool): スリッページを含めるか
        slippage_rate (float): スリッページ率（デフォルト0.01%）
        
    Returns:
        Tuple[int, float, float, float]: 
            - 購入可能株数
            - 約定代金
            - 手数料
            - 総コスト（約定代金+手数料+スリッページ）
            
    Example:
        >>> qty, value, comm, total = calculate_max_affordable_quantity(
        ...     available_funds=1000000,
        ...     stock_price=3216.36
        ... )
        >>> print(f"{qty}株, 総コスト={total:.0f}円")
        300株, 総コスト=965443円
    """
    if available_funds <= 0 or stock_price <= 0:
        return 0, 0.0, 0.0, 0.0
    
    # 理論上の最大株数
    max_possible = int(available_funds / stock_price)
    
    # 単元株数に調整（切り捨て）
    max_possible = (max_possible // unit_size) * unit_size
    
    if max_possible == 0:
        logger.warning(
            f"資金不足: 1単元({unit_size}株)も購入できません。"
            f"必要額={stock_price * unit_size:.0f}円, 利用可能={available_funds:.0f}円"
        )
        return 0, 0.0, 0.0, 0.0
    
    # 二分探索で最適な株数を探す
    best_quantity = 0
    best_contract_value = 0.0
    best_commission = 0.0
    best_total_cost = 0.0
    
    for qty in range(unit_size, max_possible + 1, unit_size):
        contract_value = qty * stock_price
        commission = calculate_japanese_stock_commission(contract_value)
        
        # スリッページ計算
        slippage = contract_value * slippage_rate if include_slippage else 0.0
        
        # 総コスト
        total_cost = contract_value + commission + slippage
        
        if total_cost <= available_funds:
            best_quantity = qty
            best_contract_value = contract_value
            best_commission = commission
            best_total_cost = total_cost
        else:
            break  # 予算超過
    
    if best_quantity == 0:
        logger.warning(
            f"手数料込みで購入不可: 株価={stock_price:.2f}円, "
            f"利用可能資金={available_funds:.0f}円"
        )
    
    return best_quantity, best_contract_value, best_commission, best_total_cost


def calculate_total_trade_cost(
    quantity: int,
    stock_price: float,
    include_slippage: bool = True,
    slippage_rate: float = 0.0001
) -> Tuple[float, float, float, float]:
    """
    取引の総コストを計算（約定代金+手数料+スリッページ）
    
    Args:
        quantity (int): 株数
        stock_price (float): 株価（円）
        include_slippage (bool): スリッページを含めるか
        slippage_rate (float): スリッページ率（デフォルト0.01%）
        
    Returns:
        Tuple[float, float, float, float]:
            - 約定代金
            - 手数料
            - スリッページ
            - 総コスト
            
    Example:
        >>> value, comm, slip, total = calculate_total_trade_cost(
        ...     quantity=100,
        ...     stock_price=3216.36
        ... )
        >>> print(f"約定代金={value:.0f}円, 手数料={comm:.0f}円, 総コスト={total:.0f}円")
        約定代金=321636円, 手数料=275円, 総コスト=321943円
    """
    if quantity <= 0 or stock_price <= 0:
        return 0.0, 0.0, 0.0, 0.0
    
    contract_value = quantity * stock_price
    commission = calculate_japanese_stock_commission(contract_value)
    slippage = contract_value * slippage_rate if include_slippage else 0.0
    total_cost = contract_value + commission + slippage
    
    return contract_value, commission, slippage, total_cost


# ユーティリティ関数
def format_commission_report(
    quantity: int,
    stock_price: float,
    symbol: str = ""
) -> str:
    """
    手数料レポートを整形して返す（ログ出力用）
    
    Args:
        quantity (int): 株数
        stock_price (float): 株価
        symbol (str): 銘柄コード（オプション）
        
    Returns:
        str: 整形されたレポート文字列
    """
    contract_value, commission, slippage, total_cost = calculate_total_trade_cost(
        quantity, stock_price
    )
    
    symbol_str = f"{symbol} " if symbol else ""
    
    return (
        f"【取引コスト】{symbol_str}{quantity}株 @ {stock_price:.2f}円\n"
        f"  約定代金: {contract_value:,.0f}円\n"
        f"  手数料: {commission:,.0f}円\n"
        f"  スリッページ: {slippage:,.2f}円\n"
        f"  総コスト: {total_cost:,.0f}円"
    )


if __name__ == "__main__":
    # テストコード
    print("=== 三菱UFJ eスマート証券 手数料計算テスト ===\n")
    
    # テスト1: 各価格帯の手数料
    test_values = [45000, 75000, 150000, 350000, 900000, 5000000]
    print("【テスト1】約定代金別の手数料")
    for value in test_values:
        comm = calculate_japanese_stock_commission(value)
        print(f"  約定代金 {value:,}円 → 手数料 {comm:.0f}円")
    
    # テスト2: 単元株調整
    print("\n【テスト2】単元株調整")
    test_quantities = [50, 150, 250, 99, 101]
    for qty in test_quantities:
        adjusted_up = adjust_to_trading_unit(qty, round_up=True)
        adjusted_down = adjust_to_trading_unit(qty, round_up=False)
        print(f"  {qty}株 → 切り上げ: {adjusted_up}株, 切り捨て: {adjusted_down}株")
    
    # テスト3: 購入可能株数計算
    print("\n【テスト3】購入可能株数計算（トヨタ想定）")
    available = 1000000  # 100万円
    price = 3216.36
    qty, value, comm, total = calculate_max_affordable_quantity(available, price)
    print(f"  利用可能資金: {available:,}円")
    print(f"  株価: {price:.2f}円")
    print(f"  購入可能株数: {qty}株")
    print(f"  約定代金: {value:,.0f}円")
    print(f"  手数料: {comm:.0f}円")
    print(f"  総コスト: {total:,.0f}円")
    print(f"  残金: {available - total:,.0f}円")
    
    # テスト4: レポート出力
    print("\n【テスト4】レポート出力")
    print(format_commission_report(300, 3216.36, "7203.T"))
