# Copilot Instructions - Concise Version

## 🎯 **Core Principles**
1. **Actual Backtest Execution Required**: Never skip `strategy.backtest()` calls
2. **Signal Generation Mandatory**: Always produce `Entry_Signal`/`Exit_Signal` columns
3. **No Fabricated Responses**: If unsure, say "I don't know" instead of making things up

## 📋 **Response Quality Rules**
- **Verify before claiming**: Test actual execution, check actual numbers
- **If output shows profit=0**: Investigate cause, don't say "perfect"
- **Excel output prohibited**: Use CSV+JSON+TXT instead (since 2025-10-08)

## 🔧 **System Architecture**
- Main entry: `main.py` in project root
- Strategies: `strategies/*.py` with `backtest()` method
- Output: `output/unified_exporter.py` (no Excel)

## ⚠️ **Known Issues**
- Unicode characters cause Windows terminal errors
- Two `main.py` files exist (root works, src/main.py has path issues)
- Text output may show profit=0 even when system runs

## 🚨 **Mandatory Checks**
- Always validate actual trade count > 0
- Verify signal columns exist
- Check output file contents, not just file existence
- Report exact numbers, not assumptions

---
**Remember: This project exists to run backtests. Any change that prevents actual backtest execution violates the core purpose.**
