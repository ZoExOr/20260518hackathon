# Strategy Guide

You are running a restaurant, not optimizing a spreadsheet. The simulation rewards nuance.

This guide gives you the vocabulary to think about the problem. It does not give you the answers. Teams that analyze their data will outperform teams that guess.

---

## The Five Tensions

Every decision creates tension between competing goals. There is no free lunch.

### 1. Profit vs. Quality

Cutting staff saves 120 EUR per person per day. But fewer staff means slower kitchens, longer waits, and walkouts. The savings show up in your cash balance; the damage shows up in your reputation — later, and harder to reverse.

### 2. Short-Term vs. Long-Term

Slashing costs produces instant savings. But tomorrow's customers are influenced by yesterday's decisions. Reputation has momentum — it is harder to recover than to maintain. A restaurant that tanks its reputation early spends the rest of the month digging out.

### 3. Specialization vs. Resilience

Buying from one supplier is simple and sometimes cheaper. But supply chains are fragile. Concentrating your orders makes you vulnerable to disruptions you cannot predict. Diversifying costs more but protects against shocks. Suppliers that seem reliable today may not be reliable tomorrow.

### 4. Cost vs. Coverage

Understaffing saves money on wages. Overstaffing wastes money on idle hands. But understaffing also means slower kitchen throughput, longer wait times, and customers who leave before they eat. The balance point shifts with demand — Friday night needs more staff than Monday lunch.

### 5. Exploration vs. Exploitation

A large, diverse menu attracts variety seekers. But new dishes have learning curves — the kitchen is less efficient with unfamiliar recipes for the first couple of days. Shrinking to a few proven dishes feels safe but limits your customer base. Customers notice a shrinking menu.

---

## What You Can See

Your observation gives you more information than you might think at first glance.

**Exact data you can rely on:**
- Cash balance, inventory quantities (batch-level with expiry dates), pending orders
- Day-of-week patterns — track your own data across the week to find the rhythm
- Weather (today is exact; forecasts degrade over 3 days: 85%, 70%, 55%)
- Hourly covers, dishes sold, cost breakdowns
- Supplier catalogs with prices and delivery schedules
- Delivery history — ordered vs. received quantities tells a story

**Signals you must interpret:**
- Walkout counts are approximate (bands: None/Few/Some/Many)
- Reputation is shown as a band (Poor through Excellent), not a precise score
- Customer trend ("Growing", "Stable", "Declining") is directional
- Reviews are delayed — today's reviews describe visits from 1-4 days ago
- Ghost reviews come from walkouts and are always negative

**What you cannot see:**
- True demand before customers arrive (demand is censored by your capacity and inventory)
- Per-customer satisfaction scores
- Internal supplier reliability state
- Exact cohort sizes (regulars, occasional visitors, prospects)
- How the scoring formula's exact coefficients work

The information gap is the challenge. Bridge it with data analysis, pattern recognition, and strategic use of your notes field as persistent memory.

---

## Hidden Dynamics

The simulation has depth beyond what is immediately visible. Directional hints — not formulas, not exact parameters.

- **Supplier reliability is not constant.** Disruptions happen. Some suppliers are more prone to problems than others. Watch your delivery history: ordered vs. received tells a story. Concentrating too much volume with one supplier can backfire.

- **Reputation has momentum.** It is a moving average, not a simple one. Bad experiences weigh more heavily than good ones. Recovery is slow; damage is fast. And your final reputation matters more than your average reputation — don't tank quality in the last few days thinking it won't count.

- **Not all first-timers come back.** Customer pools migrate based on their experience. Regulars are worth far more than prospects (higher visit rates, higher spend). Losing regulars triggers a death spiral. One bad day can erode your customer base faster than a good week can rebuild it.

- **Variety matters.** Customers notice a shrinking menu. A diverse offering attracts a broader audience. A narrow menu may work short-term but costs you demand over time.

- **Price elasticity is real.** Raising prices reduces the number of customers willing to order. The relationship is not linear — moderate increases are tolerable, aggressive increases drive people to cheaper dishes or away entirely.

- **Promotions have diminishing returns.** Running happy hours and marketing campaigns boosts demand, but effectiveness decays with consecutive use. Stopping abruptly after sustained happy hours can create a demand dip — customers came to expect the discount.

- **Overstocking accelerates spoilage.** Hoarding ingredients beyond what you can use doesn't just tie up cash — overcrowded storage causes faster degradation of your oldest stock.

- **Staff affect everything downstream.** Kitchen speed, customer wait times, satisfaction scores. The relationship between staff level and output is not purely linear.

- **Weather isn't just flavor text.** It affects how many customers walk through your door. The forecast gives you a planning window — use it.

- **Weekday patterns exist.** Not every day is created equal. Track your own data to find the rhythm.

- **Marketing is a lever, not a switch.** Spending money on marketing can drive demand, but the relationship isn't simple. Experiment with timing and amounts — constant spend is not the same as strategic spend.

- **The full price range is worth exploring.** You can set prices between 0.8x and 1.2x base. The demand response is not symmetric — experiment to find what works.

- **Daily specials do more than you'd think.** They're not just a marketing gimmick.

- **Satisfaction is multi-dimensional.** The full dining experience matters — from the wait for a table to what ends up on the plate. Think about what makes a real customer happy or unhappy.

---

## Progressive Complexity

Think of the challenge as four layers. Master each one before reaching for the next.

### Level 1: Survive

Do not go bankrupt. This sounds trivial but the do-nothing baseline goes bankrupt by day 14. Understand your cost structure: fixed costs (300/day), staff costs (120/person/day), ingredient costs. Revenue must exceed burn rate. If cash is declining for three straight days, act immediately.

### Level 2: Optimize

Once you can survive 30 days, maximize your score. Profit is important but not everything — customer satisfaction and reputation contribute to your score too. Find the operating point that balances cost efficiency with service quality. Monitor waste rates; excessive spoilage is penalized.

Not all scoring components are created equal. Some penalties are quadratic (small gaps cost little, moderate gaps are devastating), others are linear. Figure out which is which and prioritize accordingly.

### Level 3: Adapt

The evaluation runs your agent across multiple scenarios. Some you know about; some you do not. A strategy tuned perfectly for one scenario may fail under different conditions. Build resilience: detect changing conditions from your observation data and adjust. Supply disruptions, demand surges, cost inflation — your agent needs to handle what it hasn't seen before.

### Level 4: Anticipate

The best agents don't just react — they predict. Weather forecasts hint at future demand. Supplier alerts warn of disruptions before they happen. Delivery history reveals reliability patterns. Review trends signal reputation trajectory. Day-of-week patterns are consistent. Use your notes field as persistent memory to track patterns across days.

---

## What the Leaderboard Tests

You will be evaluated across multiple scenarios — some known, some hidden. Robustness across environments matters more than perfecting one scenario.

The final evaluation runs your agent against a mix of known and hidden scenarios. Each run contributes equally to your aggregate score. An agent that performs well across diverse conditions will outrank one that excels in one scenario but collapses in others.

Consistency is a strategy. Adaptability is a superpower.
