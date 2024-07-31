只有train.py和test.py是有用的，train那就不用管了已经train完了，这两个文件对数据结构的格式要求如下
-t5
--.idea

--ckr2（字幕文件）

--t-5
---解压后得到基础模型

--trained_model
---解压后得到训练后模型

--ckrval.json（没用）
--newsub.json（没用）
--t5.py（没用）
--test.py（预测用程序，需要对该程序进行整合）
--train.py（训练程序）
--train.json 

PS：test中完成了对所有例子的分析，现在需要根据bert模型的输出结果来选择测试的例子，可能会有点麻烦。
程序最后一部分有些冗长，可以看到这段代码出现了两次
    print(f"问题: {current_question}")
    print(f"真实答案: {example['answer_start']} - {example['answer_end']}")
    print(f"预测答案: {merged_prediction}")
    print(f"最相似的字幕时间: {most_similar_subtitle[0]} - {most_similar_subtitle[1]}")
    print(f"最相似的字幕内容: {most_similar_subtitle[2]}")
    print("-" * 50)
这是因为在之前出现了一个问题，有点复杂我就不说了，反正就是用了这样比较蠢的方法用更多冗余解决了这个问题，因为代码能跑所以我就没对这段代码做优化。	
	