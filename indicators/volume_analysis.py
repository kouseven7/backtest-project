def detect_volume_increase(current_volume: float, previous_volume: float, threshold: float = 1.2) -> bool:
    """
    出来高の増加を判定する。

    Parameters:
        current_volume (float): 現在の出来高
        previous_volume (float): 前日の出来高
        threshold (float): 増加率の閾値（デフォルトは1.2）

    Returns:
        bool: 出来高が閾値を超えて増加している場合は True、それ以外は False
    """
    return current_volume > previous_volume * threshold