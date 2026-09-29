# LightGBM 竞赛训练代码

本仓库保存比赛训练器、金融逻辑特征定义、滚动验证代码、模型配置和研究说明。

数据与生成结果不进入 Git：题目数据位于 `D:\competition\赛题五数据`，模型、运行目录、日志和报告文件按 `.gitignore` 排除。这样仓库只保存可复现训练所需的源码与配置，不会把大型数据文件提交到远程仓库。

主要入口：

- `train_models.py`：独立 LightGBM 训练入口，每次运行创建新的运行目录。
- `financial_logic.py`：金融逻辑特征登记、审核和特征目录校验。
- `walkforward.py`：扩展窗口滚动验证协议。
- `reporting.py`、`word_report.py`：训练报告生成。
- `configs/`：各代模型的 JSON 配置。

在 GitHub Desktop 中添加本地仓库：选择 File > Add local repository，目录选择本文件所在的 `D:\competition\train`。之后可以在 Changes 页面查看、提交并发布到 GitHub。
