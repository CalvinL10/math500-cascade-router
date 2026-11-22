# MATH 550 Final Project — Zero-Decimal Stage Cue Cards (`Slides 1–10`)

> **Design Rule**: **ZERO decimals, ZERO hard numbers, ZERO tongue-twisters.** Only short everyday English sentences (~730 words total -> ~7–8 minutes at a calm pace, leaving ~7 minutes for Q&A inside the 15-minute slot). Exact decimals stay on the slides for the professor—you just point and tell the story!

---

## Main Deck (Slides 1–10 · Target ~7:30)

### Slide 1 · Cover: Router Helper & Better Tradeoff (`0:00 – 0:30`)
- **指屏幕**: 标题 → 小模型框 → 中间黄框 Router Helper → 右下红框 Better Tradeoff
- **中文**: 打招呼 → 主题：训练一个 Router 帮手来获得更好的速度-准确率折中 → 小模型先答，帮手判断留还是转大模型 → 时间砍半，分数几乎不变。
- **Speak**:
  1. "Hi everyone, I am Zishun. // In this project, we train a simple **router helper** to get a **better speed–accuracy tradeoff** on math problems."
  2. "A fast small model tries first. // Then our router helper judges the answer: // keep it, or send it to a large model."
  3. "As shown in red, we cut waiting time **almost in half**, // with **almost the same score**."

### Slide 2 · Why Do We Need a Router? (`0:30 – 1:10`)
- **指屏幕**: 左上小模型 → 左下大模型 → 右边说明 → 底部红框问题
- **中文**: 为什么需要 Router？小的快、大的准但慢四倍；所以只数词数不公平，要看时间；引出问题：小模型自己能追上吗，还是需要 Router？
- **Speak**:
  1. "Why do we need a router helper?"
  2. "We test two fixed models. // On the left, the small model is fast, // while the large model is more accurate, // **but** runs **four times slower** per token."
  3. "Because of this speed gap, counting tokens alone is not fair—we must compare **time** too."
  4. "So at the bottom: can the small model close this gap by itself, // or do we need a router to send hard problems to the large model?"

### Slide 3 · First Check: Five Baseline Options (`1:10 – 1:55`)
- **指屏幕**: 表格第 1 行 → 第 5 行 → 底部随机性说明
- **中文**: 先看不用 Router 的五种简单做法：单试、自改、投票、静默思考、直接大模型；只有投票和思考开了随机性。
- **Speak**:
  1. "First, let us check five baseline ways without a router."
  2. "First, the small model tries once. // Second, Self-Refine checks its own answer and rewrites it."
  3. "Third, Voting runs two random paths, // and uses a third if they disagree. // Fourth, Thinking Mode thinks silently first."
  4. "Fifth, we just use the large model directly. // Only voting and thinking use randomness to try different paths."

### Slide 4 · Why Trying Harder Inside the Small Model Fails (`1:55 – 2:40`)
- **指屏幕**: 左图绿红棕柱 → 右边 3 条加粗标题 → 右下红框过渡句
- **中文**: 小模型自己硬卷不行：自改改不对错题、思考近半写爆掉分、投票太慢 → 顺滑引出：所以我们需要一个 Router 帮手！
- **Speak**:
  1. "So what happens when the small model tries harder by itself? // Green is right, red is wrong, and brown means cut off."
  2. "Self-Refine **cannot** fix its own finished mistakes. // Thinking Mode **drops** in score because almost **half** get cut off—shown by the tall brown bars."
  3. "Voting helps, // **but** still takes too long."
  4. "So staying inside the small model hits a wall. // Next, let us build our **router helper** to send hard problems to the big model."

### Slide 5 · Our Router Helper: Two-Step Decision Rule (`2:40 – 3:30`)
- **指屏幕**: 流程图黄框 → 红框 → 中间公式 → 底部第 3 列引出下一页
- **中文**: 我们的 Router 怎么判断？一步：写爆必转；二步：写完了问“大模型能救多少”减“值不值得等”，比只看心虚的 ConfGate 聪明；下页讲怎么提前预测这三个数。
- **Speak**:
  1. "Here is how our router helper judges when to ask the big model."
  2. "Step one, the yellow box: if the small model gets **cut off**, we **always** send it to the big model, because cut-off answers get zero."
  3. "Step two, the red box: if it finishes, we check the middle formula."
  4. "It asks two simple questions: // **can the big model fix it**, // and **is it worth the wait**?"
  5. "Unlike Confidence Gate, we check **both** gain and wait time. // Next, how do we predict those scores **before** running the big model?"

### Slide 6 · How We Train the Helper: Clues & Class Models (`3:30 – 4:20`)
- **指屏幕**: 顶部三份数据 → 左边 6 个实时线索 → 右边 3 个课上简单模型
- **中文**: 怎么提前预测并训练 Router？三份数据防作弊；只看小模型刚做完时的 6 个实时线索（不看难度）；用 2 个逻辑回归 + 1 个岭回归；线性比树更稳。
- **Speak**:
  1. "To make that judgment right after the small model finishes, we train our helper without cheating."
  2. "At the top, we split the data into training, // validation to lock settings, // and test **only for reporting**."
  3. "On the left, we grab **six** real-time clues—like confidence and answer length—with problem difficulty hidden."
  4. "On the right, we put them into **three** simple linear models from class: // two predict if each model is right, // and one predicts big-model wait time."
  5. "Simple linear models match tree models here, and are safer."

### Slide 7 · Main Results: A Better Tradeoff in Time (`4:20 – 5:10`)
- **指屏幕**: 右图蓝+ → 左图红线 → 左图红线 vs 蓝线 → 底部 3 个粗体大字
- **中文**: 看结果有没有更好折中！高且靠左好。右图 token 输给直接大模型；左图因为小模型快四倍，红线甩到最左砍半时间，也比蓝线 ConfGate 快。
- **Speak**:
  1. "Now, did our router helper actually get a **better tradeoff**? // Higher and farther left is better."
  2. "Look at the **right plot** first for raw tokens. // Here the blue plus—the big model alone—wins, // because we always run the small model first."
  3. "Now look at the **left plot** for estimated time. // Because the small model is much faster, the ranking flips!"
  4. "Our red line cuts waiting time **almost in half** compared to the big model, // with only a small drop in score."
  5. "At the same switches, we are also faster and higher than the blue Confidence Gate."

### Slide 8 · Why Gains Are Modest & Smart Skip Example (`5:10 – 6:05`)
- **指屏幕**: 左图 1→10 组 → 第 1 组大红跳 → 第 2 组小跳大灰柱 → 右下蓝框例子
- **中文**: 为什么没涨更高？分 10 组看：第 1 组写爆大赚；第 2 组不赚还巨慢；猜救回很难；右下例子跳过省时间。
- **Speak**:
  1. "Why isn't the gain larger? // Here we split test problems into **ten** groups, from least to most confident."
  2. "Black is the small model, red is the big model, // and gray bars show how long the big model writes."
  3. "In **group one**, cut-off answers get a **huge red jump**. // In **group two**, the red jump is tiny, while the gray bar is huge."
  4. "Our clues spot small-model errors well, // **but** it is hard to guess what the big model can fix."
  5. "In the bottom-right example, Confidence Gate sends it. // Our helper sees the big model will also fail slowly—so it **skips** it and saves time."

### Slide 9 · Statistical Checks (`6:05 – 6:55`)
- **指屏幕**: 左图重叠误差棒 → 右图红色 4 条 → 右图蓝灰条
- **中文**: 左图重叠，右图同题配对消除难度噪音；校正后四条红条实打实赢小模型基线；蓝灰条太近不分胜负，不吹追平大模型。
- **Speak**:
  1. "Next, are these gains statistically real?"
  2. "On the left, error bars overlap. // On the right, we compare on the **same problems**, which removes difficulty noise."
  3. "After multiple testing checks, the **four red bars** show real wins over all small-model baselines, // including only sending cut-off answers."
  4. "The blue and gray bars are **too close to call**."
  5. "So we do **not** claim we match the big model in accuracy—we trade a little accuracy for much faster speed."

### Slide 10 · Validation Check, Limits & Wrap-Up (`6:55 – 7:45`)
- **指屏幕**: 左表 + 蓝框 → 右边 3 条总结
- **中文**: 诚实交代：救写爆与省时间两边都稳；写完排序加分仅见于测试集；总结两句致谢。
- **Speak**:
  1. "Finally, let us check what holds across datasets."
  2. "On the left, two results hold on **both** validation and test: // rescuing cut-off answers, // and saving time over Confidence Gate."
  3. "However, the extra gain on finished answers appeared **only on test**, // because the two models are close and hit the same length limit."
  4. "To wrap up: // send cut-off answers first, // and route by gain minus wait time for a better tradeoff. // Thank you—any questions?"

---

## Q&A Easy Replies (Zero Decimals — Click Backup Slide & Speak Plain English)

- **Q1 (Why didn't Tier-2 lift Validation? How did you pick $\lambda$?) → Click `B1` / `B2`**:
  - "On validation, Tier two gave zero extra gain; that extra gain only appeared on test, and we report that honestly. Across our penalty grid, test accuracy stays almost flat—so we do not claim one best lambda."
- **Q2 (Why do Fig 4A bars overlap when 4B is significant?) → Stay on `Slide 9`**:
  - "The left plot checks each method alone. The right plot compares them on the **same** problems, removing difficulty noise. Four red bars pass multiple testing, while voting and the big model are too close to call."
- **Q3 (What about batch serving or a different GPU?) → Click `B3`**:
  - "In large batches, our router ties voting in speed. For other GPUs, Backup B3 shows a shaded pink band of speed ratios where our router is fastest—and our GPU sits right inside that band."
- **Q4 (How did you calibrate and check grading?) → Click `B4`**:
  - "We picked calibration by lowest error on validation, before seeing test. We also fixed two formatting bugs in the math grader across all outputs, and our main findings stayed the same."

### Emergency Rescue Sentences (If You Freeze on Stage)
1. `"As you can see on the slide, the red line is faster."`
2. `"The exact numbers are right here on the slide."`
3. `"Let me move to the next slide."`

### Seven Safety Rules (Never Overclaim)
1. **Tier-2 lift**: Say `+2.4%` appeared on **Test**; `Val` showed `0` Tier-2 gain (and Dev CV was near random).
2. **vs. 8B Direct**: Trade-off (`−3.0%` acc, `−40.5%` est. latency) — never say "equivalent" or "non-inferior".
3. **vs. SC / ConfGate**: Not significant in accuracy after Holm; cite `−5.6 s` latency savings at matched `k = 123`.
4. **Amber random lines**: Use **unadjusted** bootstrap intervals.
5. **Lambda ($\lambda$)**: Flat sensitivity (`75.0%–75.6%`) — never say optimal.
6. **Latency**: Always **estimated serial latency** (excludes queueing/switching), and acknowledge `+36.7%` raw tokens.
7. **Splits**: `MATH-500` **used only for reporting in the current pipeline**.
