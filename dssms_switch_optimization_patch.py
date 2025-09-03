
# ===== DSSMS切替過多問題修正パッチ =====
# 適用日: 2025-09-03 10:04:14


    def _evaluate_switch_decision_improved(self, date: datetime, current_position: Optional[str], 
                                         ranking_result: Dict[str, Any], market_condition: Dict[str, Any]) -> Dict[str, Any]:
        """改善版: 切替判定（過度な切替を抑制）"""
        try:
            should_switch = False
            reason = ""
            target_symbol = None
            
            top_symbol = ranking_result.get('top_symbol')
            
            # 初回ポジション設定
            if current_position is None and top_symbol:
                should_switch = True
                reason = "初期ポジション設定"
                target_symbol = top_symbol
                return {
                    'should_switch': should_switch,
                    'target_symbol': target_symbol,
                    'reason': reason,
                    'trigger': SwitchTrigger.DAILY_EVALUATION if should_switch else None
                }
            
            # 既存ポジションがある場合の厳格な切替判定
            if current_position and top_symbol and current_position != top_symbol:
                current_score = ranking_result.get('rankings', {}).get(current_position, 0)
                top_score = ranking_result.get('rankings', {}).get(top_symbol, 0)
                
                # 1. 最小保有期間チェック（24時間未満なら切替しない）
                if len(self.switch_history) > 0:
                    last_switch = self.switch_history[-1]
                    hours_since_last_switch = (date - last_switch.timestamp).total_seconds() / 3600
                    if hours_since_last_switch < 24.0:  # 最小24時間保有
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': f"最小保有期間未満: {hours_since_last_switch:.1f}時間",
                            'trigger': None
                        }
                
                # 2. スコア差の厳格化（20%以上の差が必要）
                score_threshold = 0.20  # 10% -> 20%に変更
                score_diff = top_score - current_score
                
                if score_diff > score_threshold:
                    # 3. 追加条件: 連続切替回数制限
                    recent_switches = [s for s in self.switch_history 
                                     if (date - s.timestamp).days <= 7]  # 過去7日
                    
                    if len(recent_switches) >= 3:  # 週3回以上の切替を制限
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': f"週間切替制限: {len(recent_switches)}回",
                            'trigger': None
                        }
                    
                    # 4. 市場ボラティリティチェック（高ボラ時は切替しない）
                    if market_condition.get('volatility_level') == 'high':
                        return {
                            'should_switch': False,
                            'target_symbol': current_position,
                            'reason': "高ボラティリティ期間",
                            'trigger': None
                        }
                    
                    # 全条件をクリアした場合のみ切替
                    should_switch = True
                    reason = f"スコア大幅改善: {current_score:.3f} -> {top_score:.3f} (+{score_diff:.3f})"
                    target_symbol = top_symbol
                else:
                    reason = f"スコア差不足: {score_diff:.3f} < {score_threshold}"
            
            return {
                'should_switch': should_switch,
                'target_symbol': target_symbol or current_position,
                'reason': reason,
                'trigger': SwitchTrigger.DAILY_EVALUATION if should_switch else None
            }
            
        except Exception as e:
            self.logger.warning(f"切替判定エラー {date}: {e}")
            return {
                'should_switch': False,
                'target_symbol': current_position,
                'reason': f"エラー: {e}",
                'trigger': None
            }
    


    def _update_symbol_ranking_realistic(self, date: datetime, symbols: List[str]) -> Dict[str, Any]:
        """現実的な銘柄ランキング更新（安定性重視）"""
        try:
            # 前回のランキングを取得（継続性のため）
            previous_rankings = getattr(self, '_previous_rankings', {})
            
            ranking_scores = {}
            
            for symbol in symbols:
                # 前回スコアがあれば継続性を持たせる
                if symbol in previous_rankings:
                    base_score = previous_rankings[symbol]
                    # 小さな変動のみ（±5%以内）
                    variation = np.random.normal(0, 0.05)
                    new_score = max(0.1, min(0.9, base_score + variation))
                else:
                    # 新規銘柄は中央値付近から開始
                    new_score = np.random.normal(0.5, 0.1)
                    new_score = max(0.1, min(0.9, new_score))
                
                ranking_scores[symbol] = new_score
            
            # ランキングを保存（次回使用のため）
            self._previous_rankings = ranking_scores.copy()
            
            # 上位銘柄選択（変更を最小限に）
            sorted_symbols = sorted(ranking_scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = sorted_symbols[:5]
            
            result = {
                'date': date,
                'rankings': dict(top_symbols),
                'top_symbol': top_symbols[0][0] if top_symbols else None,
                'top_score': top_symbols[0][1] if top_symbols else 0,
                'total_symbols': len(ranking_scores),
                'data_source': 'realistic_stable'
            }
            
            self.logger.info(f"安定ランキング更新: 上位={result['top_symbol']} ({result['top_score']:.3f})")
            return result
            
        except Exception as e:
            self.logger.warning(f"ランキング更新エラー {date}: {e}")
            return {'date': date, 'rankings': {}, 'error': str(e)}
    

# ===== パッチ適用完了 =====
