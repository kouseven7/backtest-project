"""
最適化されたパラメータを管理するモジュール
"""
import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional

class OptimizedParameterManager:
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.join(os.path.dirname(__file__), "optimized_params")
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def save_optimized_params(self, strategy_name: str, ticker: str, 
                            params: Dict[str, Any], metrics: Dict[str, float],
                            optimization_date: str = None, status: str = "pending_review",
                            validation_info: Dict = None) -> str:
        """最適化されたパラメータを保存"""
        if optimization_date is None:
            optimization_date = datetime.now().strftime("%Y-%m-%d")
        
        config = {
            "strategy": strategy_name,
            "ticker": ticker,
            "optimization_date": optimization_date,
            "parameters": params,
            "performance_metrics": metrics,
            "optimization_details": {
                "data_period": {
                    "start_date": "2020-01-01", 
                    "end_date": "2024-12-31"
                },
                "optimization_method": "grid_search",
                "cross_validation": "walk_forward",
                "total_combinations_tested": len(params) if params else 0
            },
            "status": status,
            "created_at": datetime.now().isoformat(),
            "notes": "",
            "approval_info": {
                "approved_by": None,
                "approved_at": None,
                "rejection_reason": None
            },
            "validation_info": validation_info or {}
        }
        
        filename = f"{strategy_name}_{ticker}_{optimization_date}.json"
        filepath = os.path.join(self.config_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def list_available_configs(self, strategy_name: str = None, ticker: str = None, 
                              status: str = None) -> List[Dict]:
        """利用可能な設定ファイルを一覧表示"""
        configs = []
        
        if not os.path.exists(self.config_dir):
            return configs
        
        for filename in os.listdir(self.config_dir):
            if not filename.endswith('.json'):
                continue
                
            # フィルタリング条件をチェック
            if strategy_name and not filename.startswith(strategy_name):
                continue
            if ticker and ticker not in filename:
                continue
                
            filepath = os.path.join(self.config_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # ステータスフィルタリング
                if status and config.get("status") != status:
                    continue
                
                # ファイル情報を追加
                config['filename'] = filename
                config['filepath'] = filepath
                configs.append(config)
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: {filename} の読み込みに失敗しました: {e}")
                continue
        
        # 日付順でソート（新しい順）
        configs.sort(key=lambda x: x.get('optimization_date', ''), reverse=True)
        return configs
    
    def load_specific_config(self, filename: str) -> Dict[str, Any]:
        """特定のファイル名から設定を読み込み"""
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config.get("parameters", {})
    
    def load_approved_params(self, strategy_name: str, ticker: str = None) -> Dict[str, Any]:
        """承認済みのパラメータを読み込み"""
        # 最新の承認済みパラメータを検索
        configs = self.list_available_configs(strategy_name=strategy_name, ticker=ticker, status="approved")
        
        if not configs:
            return {}
        
        # 最新の設定を返す
        latest_config = max(configs, key=lambda x: x["optimization_date"])
        return latest_config["parameters"]
    
    def interactive_select_config(self, strategy_name: str) -> Dict[str, Any]:
        """対話形式で設定ファイルを選択"""
        configs = self.list_available_configs(strategy_name=strategy_name)
        
        if not configs:
            print(f"戦略 '{strategy_name}' の設定ファイルが見つかりません。")
            return {}
        
        print(f"\n=== {strategy_name} の利用可能な設定ファイル ===")
        for i, config in enumerate(configs, 1):
            ticker = config.get('ticker', 'N/A')
            date = config.get('optimization_date', 'N/A')
            status = config.get('status', 'N/A')
            sharpe = config.get('performance_metrics', {}).get('sharpe_ratio', 'N/A')
            print(f"{i:2d}. {config['filename']}")
            print(f"    銘柄: {ticker}, 日付: {date}, ステータス: {status}")
            print(f"    シャープレシオ: {sharpe}")
            print()
        
        while True:
            try:
                choice = input("使用する設定番号を入力してください (0=デフォルト): ")
                if choice == '0':
                    return {}
                
                index = int(choice) - 1
                if 0 <= index < len(configs):
                    selected_config = configs[index]
                    print(f"選択: {selected_config['filename']}")
                    return selected_config.get("parameters", {})
                else:
                    print("無効な番号です。再入力してください。")
            except ValueError:
                print("数値を入力してください。")
    
    def get_best_config_by_metric(self, strategy_name: str, metric: str = "sharpe_ratio", 
                                 ticker: str = None, status: str = "approved") -> Dict[str, Any]:
        """指定した指標で最高パフォーマンスの設定を取得"""
        configs = self.list_available_configs(strategy_name=strategy_name, 
                                            ticker=ticker, status=status)
        
        if not configs:
            return {}
        
        # 指定した指標で最高のものを選択
        best_config = max(configs, 
                         key=lambda x: x.get('performance_metrics', {}).get(metric, -float('inf')))
        
        print(f"最高{metric}の設定を適用: {best_config['filename']} ({metric}: {best_config.get('performance_metrics', {}).get(metric, 'N/A')})")
        return best_config.get("parameters", {})
    
    def update_config_status(self, filename: str, status: str, notes: str = "", 
                           approved_by: str = None) -> bool:
        """設定ファイルのステータスを更新"""
        filepath = os.path.join(self.config_dir, filename)
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config['status'] = status
            config['notes'] = notes
            
            if status == "approved" and approved_by:
                config['approval_info']['approved_by'] = approved_by
                config['approval_info']['approved_at'] = datetime.now().isoformat()
            elif status == "rejected":
                config['approval_info']['rejection_reason'] = notes
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"ステータス更新エラー: {e}")
            return False
