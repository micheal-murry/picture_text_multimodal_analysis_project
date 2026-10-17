# 第八步：递进式回归建模日志

[INFO] === 2. 读取数据 ===
[OK] 读取成功，共 4129 行 × 133 列
[INFO] === 3. 变量检查与预处理 ===
[OK] 已构建 image_title_fit_centered（均值中心化，原均值=0.4439）
[OK] 【因变量】全部 2 个变量已就绪
[OK] 【连续控制变量】全部 7 个变量已就绪
[OK] 【分类控制变量】全部 4 个变量已就绪
[OK] 【标题特征变量】全部 7 个变量已就绪
[OK] 【封面特征（核心）】全部 7 个变量已就绪
[OK] 【封面特征（可选）】全部 2 个变量已就绪
[OK] 【双模态协同变量】全部 5 个变量已就绪
[OK] 【分类双模态变量】全部 1 个变量已就绪
[OK] 【交互效应变量】全部 1 个变量已就绪
[WARN]   封面可选变量 cover_has_chart_or_formula 全部缺失（4129 个空值），从 Model 2 中排除
[WARN]   封面可选变量 cover_is_slide_or_screenshot 全部缺失（4129 个空值），从 Model 2 中排除
[OK] 已对 14 个连续变量完成标准化（后缀 _std）
[OK] uid 列存在，共 68 个唯一UP主，将使用聚类稳健SE。
[OK] 删除关键变量缺失行：0 行，剩余 4129 行
[OK] complementarity_type 参照组设为 low_relatedness
[OK] domain 参照组设为 科学科普
[OK] level 参照组设为 0（中等层级）
[INFO] === 4. 构建回归公式 ===
[INFO] 
  [log_view] 模型公式：
[INFO]     Model0: log_view ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domain) + C(l...
[INFO]     Model1: log_view ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domain) + C(l...
[INFO]     Model2: log_view ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domain) + C(l...
[INFO]     Model3: log_view ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domain) + C(l...
[INFO]     Model4: log_view ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domain) + C(l...
[INFO] 
  [log_view_per_day] 模型公式：
[INFO]     Model0: log_view_per_day ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domai...
[INFO]     Model1: log_view_per_day ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domai...
[INFO]     Model2: log_view_per_day ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domai...
[INFO]     Model3: log_view_per_day ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domai...
[INFO]     Model4: log_view_per_day ~ log_fans + log_age + log_duration + tag_count + description_len + is_collection + copyright + C(domai...
[INFO] === 5. 运行回归模型 ===
[OK]   [log_view] Model0: N=4129, R²=0.6251, adj-R²=0.6224
[OK]   [log_view] Model1: N=4129, R²=0.6271, adj-R²=0.6238
[OK]   [log_view] Model2: N=4129, R²=0.6307, adj-R²=0.6268
[OK]   [log_view] Model3: N=4129, R²=0.6340, adj-R²=0.6293
[OK]   [log_view] Model4: N=4129, R²=0.6363, adj-R²=0.6310
[OK]   [log_view_per_day] Model0: N=4129, R²=0.6100, adj-R²=0.6072
[OK]   [log_view_per_day] Model1: N=4129, R²=0.6121, adj-R²=0.6087
[OK]   [log_view_per_day] Model2: N=4129, R²=0.6158, adj-R²=0.6118
[OK]   [log_view_per_day] Model3: N=4129, R²=0.6192, adj-R²=0.6144
[OK]   [log_view_per_day] Model4: N=4129, R²=0.6216, adj-R²=0.6161
[INFO] === 6. 整理回归结果表 ===
[INFO] === 7. 构建论文格式汇总表 ===
[INFO] === 8. 保存 Excel 输出 ===
[OK]   写入 Sheet: Wide_log_view
[OK]   写入 Sheet: Wide_log_view_per_da
[OK]   写入 Sheet: Long_log_view
[OK]   写入 Sheet: Long_log_view_per_da
[OK]   写入 Sheet: Model_Stats
[OK]   写入 Sheet: 变量说明表
[OK] Excel 已保存 → outputs\regression_results.xlsx
[INFO] === 9. 生成 Word 回归报告 ===
[WARN] python-docx 未安装，跳过 Word 输出（No module named 'docx'）。请运行：pip install python-docx
[INFO] === 10. 绘制系数图 ===
[OK]   系数图已保存 → outputs\figures\coefficient_plot_core_variables_log_view.png
[OK]   系数图已保存 → outputs\figures\coefficient_plot_core_variables_log_view_per_day.png
[OK]   主系数图已复制 → outputs\figures\coefficient_plot_core_variables.png
[OK]   R²递增图已保存 → outputs\figures\model_r2_progression.png
[INFO] === 11. 质量检查报告 ===
[INFO] 
建模样本量：4129
[INFO] 聚类单元（uid）数：68
[INFO] 
模型统计汇总（log_view）：
[INFO]   Model0: N=4129, Adj-R²=0.6224, AIC=15339.9, BIC=15529.6
[INFO]   Model1: N=4129, Adj-R²=0.6238, AIC=15332.0, BIC=15566.0
[INFO]   Model2: N=4129, Adj-R²=0.6268, AIC=15305.1, BIC=15583.4
[INFO]   Model3: N=4129, Adj-R²=0.6293, AIC=15286.9, BIC=15622.1
[INFO]   Model4: N=4129, Adj-R²=0.6310, AIC=15274.3, BIC=15653.8
[INFO] 
模型统计汇总（log_view_per_day）：
[INFO]   Model0: N=4129, Adj-R²=0.6072, AIC=15229.9, BIC=15419.7
[INFO]   Model1: N=4129, Adj-R²=0.6087, AIC=15221.2, BIC=15455.2
[INFO]   Model2: N=4129, Adj-R²=0.6118, AIC=15195.6, BIC=15473.9
[INFO]   Model3: N=4129, Adj-R²=0.6144, AIC=15177.1, BIC=15512.4
[INFO]   Model4: N=4129, Adj-R²=0.6161, AIC=15165.3, BIC=15544.8
[INFO] 
── VIF 多重共线性检查（Model 3，log_view）──
[INFO]   sentiment_conflict: VIF = inf [WARN] VIF > 10
[INFO]   log_age: VIF = 84.29 [WARN] VIF > 10
[INFO]   log_fans: VIF = 10.31 [WARN] VIF > 10
[INFO]   image_title_fit_centered_std: VIF = 4.18
[INFO]   cover_text_area_ratio_std: VIF = 2.67
[INFO]   title_cover_text_overlap_std: VIF = 2.31
[INFO]   title_question_type: VIF = 2.19
[INFO]   title_char_len_std: VIF = 2.07
[INFO]   log_duration: VIF = 1.94
[INFO]   tag_count: VIF = 1.89
[INFO]   image_title_fit_sq_centered_std: VIF = 1.86
[INFO]   face_presence: VIF = 1.83
[INFO]   cover_has_text: VIF = 1.64
[INFO]   is_collection: VIF = 1.63
[INFO]   visual_complexity_std: VIF = 1.60
[INFO]   image_colorfulness_std: VIF = 1.57
[INFO]   image_contrast_std: VIF = 1.55
[INFO]   image_brightness_std: VIF = 1.53
[INFO]   title_sentiment_intensity_std: VIF = 1.46
[INFO]   title_exclamation_type: VIF = 1.42
[INFO] === 12. 完成，保存日志 ===