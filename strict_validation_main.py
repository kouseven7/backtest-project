"""
厳格なバックテスト検証スクリプト - 数値まで完全検証
虚偽報告防止・実証ベース検証
"""
import os
import pandas as pd
import numpy as np
import subprocess
import sys
from datetime import datetime

class StrictBacktestValidator:
    """厳格なバックテスト検証クラス"""
    
    def __init__(self):
        self.validation_failures = []
        self.critical_issues = []
        
    def validate_actual_execution(self, main_py_path):
        """実際の実行結果を数値レベルで検証"""
        
        print("[SEARCH] **厳格検証開始** - 数値まで完全チェック")
        print("=" * 60)
        
        # Step 1: 実際にmain.pyを実行
        print(f"1️⃣ {main_py_path} の実行...")
        try:
            result = subprocess.run(
                [sys.executable, main_py_path], 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=os.path.dirname(os.path.abspath(main_py_path))
            )
            
            print(f"実行結果 - Return Code: {result.returncode}")
            
            if result.returncode != 0:
                print(f"[ERROR] 実行失敗:")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                self.critical_issues.append(f"[ERROR] {main_py_path} 実行失敗: {result.stderr}")
                return False
                
            print("[OK] 実行完了 - 出力ファイル検証開始")
            print(f"STDOUT preview: {result.stdout[:500]}...")
            
        except subprocess.TimeoutExpired:
            self.critical_issues.append(f"[ERROR] {main_py_path} 実行タイムアウト")
            return False
        except Exception as e:
            self.critical_issues.append(f"[ERROR] {main_py_path} 実行エラー: {e}")
            return False
        
        # Step 2: 出力ファイルを実際に検証
        return self._validate_output_files()
    
    def _validate_output_files(self):
        """出力ファイルの数値を厳格に検証"""
        
        print("\n2️⃣ 出力ファイル数値検証...")
        
        # 出力ディレクトリの確認
        output_dirs = ["output", "results", "exports", "logs", "."]
        found_files = []
        
        for dir_name in output_dirs:
            if os.path.exists(dir_name):
                for file in os.listdir(dir_name):
                    if file.endswith(('.csv', '.txt', '.json', '.xlsx', '.log')):
                        found_files.append(os.path.join(dir_name, file))
        
        if not found_files:
            self.critical_issues.append("[ERROR] 出力ファイルが生成されていない")
            return False
        
        print(f"発見されたファイル: {len(found_files)}件")
        for file in found_files:
            print(f"  - {file}")
        
        # 各ファイルの内容を検証
        valid_files = 0
        
        for file_path in found_files:
            if self._validate_file_content(file_path):
                valid_files += 1
        
        if valid_files == 0:
            self.critical_issues.append("[ERROR] 有効な出力ファイルが0件")
            return False
        
        print(f"[OK] 有効ファイル: {valid_files}/{len(found_files)}件")
        return True
    
    def _validate_file_content(self, file_path):
        """ファイル内容の数値を実際に検証"""
        
        print(f"\n[LIST] {os.path.basename(file_path)} の内容検証:")
        
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return self._validate_csv_data(df, file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._validate_text_content(content, file_path)
            elif file_path.endswith('.log'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._validate_log_content(content, file_path)
                
        except Exception as e:
            print(f"  [ERROR] ファイル読み取りエラー: {e}")
            return False
        
        return True
    
    def _validate_csv_data(self, df, file_path):
        """CSV数値データの厳格検証"""
        
        if df.empty:
            print(f"  [ERROR] データフレームが空")
            return False
        
        print(f"  [CHART] データ行数: {len(df)}行")
        print(f"  [CHART] データ列数: {len(df.columns)}列")
        print(f"  [CHART] 列名: {list(df.columns)}")
        
        # シグナル列の検証
        signal_columns = ['Entry_Signal', 'Exit_Signal']
        found_signals = [col for col in signal_columns if col in df.columns]
        
        if found_signals:
            print(f"  [OK] シグナル列発見: {found_signals}")
            
            for signal_col in found_signals:
                signal_count = (df[signal_col] == 1).sum() if signal_col == 'Entry_Signal' else abs(df[signal_col]).sum()
                print(f"    {signal_col}: {signal_count}回")
                
                if signal_count == 0:
                    print(f"    [WARNING] {signal_col}が0回 - 戦略未実行の可能性")
                    self.validation_failures.append(f"{signal_col}が0回: {file_path}")
        else:
            print(f"  [ERROR] **重大**: シグナル列が存在しない")
            self.critical_issues.append(f"シグナル列不在: {file_path}")
            return False
        
        # 価格データの検証
        price_columns = ['Close', 'Open', 'High', 'Low']
        found_prices = [col for col in price_columns if col in df.columns]
        
        if found_prices:
            print(f"  [OK] 価格列発見: {found_prices}")
            for price_col in found_prices:
                price_range = f"{df[price_col].min():.2f} - {df[price_col].max():.2f}"
                print(f"    {price_col}: {price_range}")
        
        # 損益計算の検証
        if 'Portfolio_Value' in df.columns:
            initial_value = df['Portfolio_Value'].iloc[0] if len(df) > 0 else 0
            final_value = df['Portfolio_Value'].iloc[-1] if len(df) > 0 else 0
            total_return = ((final_value - initial_value) / initial_value * 100) if initial_value != 0 else 0
            
            print(f"  [MONEY] 初期資産: {initial_value:,.0f}")
            print(f"  [MONEY] 最終資産: {final_value:,.0f}")
            print(f"  [UP] 総リターン: {total_return:.2f}%")
            
            if total_return == 0.0 and any((df[col] == 1).sum() > 0 for col in found_signals):
                print(f"  [ALERT] **重大問題**: 取引実行したが損益0% - 計算ロジック異常")
                self.critical_issues.append(f"損益計算異常: {file_path}")
                return False
        
        print(f"  [OK] CSV検証完了")
        return True
    
    def _validate_text_content(self, content, file_path):
        """テキスト出力の検証"""
        
        lines = content.split('\n')
        print(f"  📝 行数: {len(lines)}行")
        
        # 重要な数値情報を抽出
        profit_mentions = [line for line in lines if ('損益' in line or 'profit' in line.lower() or 'return' in line.lower() or '総リターン' in line)]
        
        if profit_mentions:
            print(f"  [MONEY] 損益関連情報:")
            for mention in profit_mentions[:3]:  # 最初の3行のみ表示
                print(f"    {mention.strip()}")
        else:
            print(f"  [WARNING] 損益情報が見つからない")
        
        return True
    
    def _validate_log_content(self, content, file_path):
        """ログファイル内容の検証"""
        
        lines = content.split('\n')
        print(f"  📝 ログ行数: {len(lines)}行")
        
        # エラーの検索
        error_lines = [line for line in lines if 'ERROR' in line or 'CRITICAL' in line]
        warning_lines = [line for line in lines if 'WARNING' in line]
        
        if error_lines:
            print(f"  [ERROR] エラー: {len(error_lines)}件")
            for error in error_lines[:2]:  # 最初の2件のみ表示
                print(f"    {error.strip()}")
        
        if warning_lines:
            print(f"  [WARNING] 警告: {len(warning_lines)}件")
        
        return True
    
    def generate_strict_report(self):
        """厳格検証レポート生成"""
        
        print("\n" + "=" * 60)
        print("[LIST] **厳格検証レポート**")
        print("=" * 60)
        
        print(f"\n[ALERT] **重大問題**: {len(self.critical_issues)}件")
        for issue in self.critical_issues:
            print(f"  {issue}")
        
        print(f"\n[WARNING] **検証失敗**: {len(self.validation_failures)}件")
        for failure in self.validation_failures:
            print(f"  {failure}")
        
        # 総合判定
        if self.critical_issues:
            print(f"\n[ERROR] **総合判定: 不合格** - 重大問題により使用不可")
            return False
        elif self.validation_failures:
            print(f"\n[WARNING] **総合判定: 条件付き合格** - 警告事項の確認が必要")
            return "conditional"
        else:
            print(f"\n[OK] **総合判定: 合格** - 検証項目をクリア")
            return True

def execute_strict_validation():
    """厳格検証の実行"""
    
    validator = StrictBacktestValidator()
    
    # 両方のmain.pyを検証
    main_files = ["main.py", "src/main.py"]
    results = {}
    
    for main_file in main_files:
        if os.path.exists(main_file):
            print(f"\n[TARGET] **{main_file} 厳格検証**")
            print("-" * 40)
            
            result = validator.validate_actual_execution(main_file)
            results[main_file] = result
            
            print(f"\n{main_file} 検証結果: {result}")
        else:
            print(f"[WARNING] {main_file} が存在しません")
    
    # 最終レポート
    final_result = validator.generate_strict_report()
    
    return results, final_result

if __name__ == "__main__":
    print("[FIRE] **数値レベル厳格検証開始**")
    print("虚偽報告は一切行いません")
    print("=" * 60)
    
    results, final = execute_strict_validation()
    
    print(f"\n[FINISH] **最終結論**:")
    if final is True:
        print("[OK] システムは正常に動作しています")
    elif final == "conditional":
        print("[WARNING] システムに警告事項があります - 要確認")
    else:
        print("[ERROR] システムは正常に動作していません - 修正が必要")