# 執行範例腳本報告

此儲存庫內的範例腳本 `example/single_spin_try_run.py` 會使用 `matplotlib` 的 Qt 圖形後端 `qtagg`。在未安裝 Qt 相依套件的環境下，直接執行會出現以下錯誤：

```
ImportError: Failed to import any of the following Qt binding modules: PyQt6, PySide6, PyQt5, PySide2
```

為在無圖形介面的環境下仍能測試運行，我們複製腳本並改用非互動式的 `Agg` 後端，檔名為 `example/single_spin_try_run_ag.py`。內容僅於第五行指定：

```python
matplotlib.use("Agg")
```

在安裝 `matplotlib` 等必要套件後，執行此改寫版本可順利完成，僅因使用 `Agg` 後端而不顯示視窗，但程式能正常產生計算結果。
