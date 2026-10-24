# Instamart Customer Intelligence & Support Copilot

## Overview

This repository is an internal Customer Intelligence & Support Copilot for a fictional Instamart-style grocery business. All data is synthetic. All amounts are in INR. The copilot is built in four parts: a trusted analytical database, customer segments and lapse scores, support-ticket triage, and a policy assistant with a refund agent.

Part A is complete. It audits the raw exports, cleans customers and orders with explicit rules, stores the result in SQLite, and answers seven business questions in SQL with a pandas cross-check. Part B is complete. It names four customer segments and scores who is likely to lapse. Part C is complete. It redacts support tickets and routes them with the offline classifier. Part D is complete. It answers policy questions from the five documents and decides refund requests against the order database.

The supplied files `generate_data.py` and `mock_llm.py` are unchanged. Parts C and D call that offline model. No network call and no API key are used.

## Reproduce

Run these commands from the repository root, in order. This machine provides `python3` (3.11 or newer is required). After the virtual environment is activated, `python` points at that environment.

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python scripts/audit_raw.py`
6. `python scripts/clean_and_store.py`
7. `python scripts/validate.py`
8. `python scripts/run_queries.py`
9. `python scripts/eda.py`
10. `python scripts/segment_customers.py`
11. `python scripts/train_lapse.py`
12. `python scripts/triage_tickets.py`
13. `python scripts/policy_assistant.py`
14. `python scripts/refund_agent.py`

`generate_data.py` must end with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. `scripts/validate.py` must print `ALL CHECKPOINTS PASSED`. `scripts/segment_customers.py` must print `SEGMENTS PASSED`. `scripts/train_lapse.py` must print `LAPSE PASSED`. `scripts/triage_tickets.py` must print `TRIAGE PASSED`. `scripts/policy_assistant.py` must print `POLICY PASSED`. `scripts/refund_agent.py` must print `AGENT PASSED`.

## Part A

### Audit

The three raw tables were loaded with pandas as strings (`dtype=str`, `keep_default_na=False`), so blank cells stay blank and the token `N/A` is preserved. Every column below has dtype `str` under that load.

| Table | Shape | Column | Missing | Missing % |
|---|---:|---|---:|---:|
| customers | 1,200 × 5 | customer_id | 0 | 0.00 |
| customers | | city | 0 | 0.00 |
| customers | | signup_date | 0 | 0.00 |
| customers | | age | 61 | 5.08 |
| customers | | gender | 30 | 2.50 |
| products | 40 × 4 | product_id, product_name, category, price_inr | 0 each | 0.00 |
| orders | 13,084 × 9 | order_id, customer_id, product_id, quantity, order_ts, status, discount_pct, order_amount | 0 each | 0.00 |
| orders | | delivery_minutes | 2,541 | 19.42 |

`order_amount` has no blank cells. The unparseable tokens `N/A` and `error` are present values and are counted in the checkpoint table, not as missing.

| Check | Value |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank gender | 30 |

### Cleaning rules

Customers, in order:

1. City: strip spaces and apply title case. This rewrote 476 values. Six cities remain: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. Gender: map `m` / `male` and `f` / `female` in any case to `male` or `female`. A blank becomes `unknown`. This rewrote 950 values, including 30 blanks set to unknown. Result: female 610, male 560, unknown 30.
3. Age: values outside 10–100 were set to missing (6 values: the planted ages). Together with 61 blank ages, 67 ages were filled with the median age of the customer's city using `groupby` and `transform`. Missing ages afterwards: 0. Every age is between 18 and 60.
4. `signup_date` was parsed as a datetime and stored as `YYYY-MM-DD`. Row count unchanged at 1,200.

Orders, in order:

1. `order_ts` was parsed as a datetime for all 13,084 rows, and is stored as text `YYYY-MM-DD HH:MM:SS`.
2. Exact duplicate rows removed: 119. Rows remaining: 12,965.
3. For each `order_id`, only the row with the latest `order_ts` was kept. This removed 656 earlier rows, which are the pending copies. Rows remaining: 12,309, one per order. Pending rows remaining: 0. Status counts: 11,110 completed and 1,199 cancelled.
4. `order_amount` was coerced to a number. The raw file has 120 unparseable amounts; 8 of those sat on rows removed by rules 2 and 3. The 112 missing amounts on the kept rows were filled with `round(price_inr × quantity × (1 − discount_pct / 100), 2)`. Missing amounts afterwards: 0.
5. Negative `delivery_minutes` were set to missing. The raw file has 106 negatives; 1 sat on a removed duplicate, so 105 kept values were cleared. Completed orders with a missing delivery time, 774 of them, were filled with the median delivery time of that product's category among completed orders, using `groupby` and `transform`. Cancelled orders were left missing, because a cancelled order was never delivered: 0 missing delivery times among completed orders, and 1,199 missing among cancelled orders.

The sum of `order_amount` over completed orders is INR 2,998,668.80.

Products needed no value changes. All 40 catalogue rows were written through to `data/clean/products.csv`.

The same three tables are in `data/instamart.db` as `customers`, `products`, and `orders`. `scripts/validate.py` checks the city counts, gender counts, age range, order counts, delivery-missing counts, and the completed-amount sum, and prints `ALL CHECKPOINTS PASSED`.

### SQL

The seven queries live in `sql/queries.sql` and are run with `pd.read_sql_query` against `data/instamart.db`. Only completed orders count as revenue or spend.

Q1. Completed orders with `order_amount` above 500, grouped by customer, kept where that total exceeds 8,000. Result: 30 customers.

Q2. Revenue by city and product category. Result: 30 rows. The 30 revenues add up to INR 2,998,668.80. The largest cell is Bengaluru / Electronics at INR 233,909.45.

| City | Dairy | Electronics | Grocery | Personal Care | Snacks |
|---|---:|---:|---:|---:|---:|
| Bengaluru | 87,075.05 | 233,909.45 | 110,915.65 | 20,193.85 | 71,549.40 |
| Chennai | 78,301.80 | 195,508.80 | 93,817.45 | 27,675.40 | 68,325.70 |
| Delhi | 117,271.70 | 158,549.10 | 112,658.05 | 23,471.50 | 90,879.45 |
| Hyderabad | 83,731.00 | 228,913.10 | 109,475.40 | 24,800.05 | 81,670.35 |
| Mumbai | 75,209.85 | 214,738.40 | 78,970.15 | 38,018.00 | 72,122.10 |
| Pune | 90,349.55 | 187,629.25 | 108,456.60 | 34,275.15 | 80,207.50 |

Q3. Products priced above the average catalogue price. Result: 11 products. Bluetooth Speaker (INR 1,999), Wireless Earbuds (INR 1,499), Power Bank 10000mAh (INR 1,299), Phone Charger 20W (INR 799), Smart Bulb (INR 599), Memory Card 64GB (INR 549), Sunscreen 50g (INR 399), Body Lotion 400ml (INR 350), Phone Stand (INR 349), Ghee 500ml (INR 340), USB-C Cable 1m (INR 299).

Q4. Customers with no completed order, using a `LEFT JOIN`. Result: 25 customers.

Q5. One CTE computes each customer's completed spend and a second computes the average of those totals. The main query returns customers above that average. Result: 550 customers. The average completed spend is INR 2,552.06.

Q6. For each customer with completed spend, `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` are computed with `OVER (PARTITION BY city ORDER BY total_spend DESC)`. The top customer in each city is Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). Exactly 8 rows have `RANK()` different from `ROW_NUMBER()`, because customers who share a spend receive the same rank while `ROW_NUMBER()` still gives each of them a different position.

Q7. Daily completed revenue, with the day taken from the first ten characters of `order_ts`, and a cumulative sum via `SUM(revenue) OVER (ORDER BY day)`. Result: 183 daily rows. The last cumulative value is INR 2,998,668.80.

### Pandas cross-check

`pd.pivot_table(..., margins=True)` on completed order amount by city and category has grand total INR 2,998,668.80, the same figure as the SQL total.

Named aggregation of completed `order_amount` by city:

| City | Sum (INR) | Mean (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Distribution of completed `order_amount`: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q3 INR 297.00. The 1.5×IQR fences are INR -283.50 and INR 645.30. There are 897 orders above the upper fence and 0 below the lower fence.

Pearson correlations on completed orders:

| Pair | Correlation |
|---|---:|
| order_amount, discount_pct | -0.095 |
| order_amount, quantity | 0.087 |
| order_amount, delivery_minutes | -0.003 |
| discount_pct, quantity | -0.000 |
| discount_pct, delivery_minutes | -0.010 |
| quantity, delivery_minutes | -0.013 |

The largest absolute correlation is `order_amount` with `discount_pct`.

These tables are six months of fictional Instamart orders, from 1 April 2025 to 30 September 2025, for 1,200 customers in six cities and 40 products across Grocery, Dairy, Snacks, Personal Care, and Electronics. The revenue base is 11,110 completed orders, totalling INR 2,998,668.80. Completed amounts are right-skewed: the mean is INR 269.91, the median is INR 135.00, and the skewness is 4.61, with quartiles at INR 64.80 and INR 297.00. Small grocery and dairy baskets sit in the centre of the distribution, and personal-care and electronics orders pull the mean up. Bengaluru's completed spend is INR 523,643.40 across 1,894 orders (mean INR 276.47); Mumbai's is INR 479,058.50 across 1,613 orders (mean INR 297.00). The 1.5×IQR fences are INR -283.50 and INR 645.30, with 897 orders above the upper fence and 0 below it. Those large orders are high catalogue prices, such as speakers and earbuds. All 1,199 cancelled orders have no delivery time, since a cancelled order was never delivered. Among amount, discount, quantity, and delivery minutes the correlations are weak. The strongest pair is order amount with discount percentage at -0.095, and order amount with quantity is +0.087. A correlation of this size does not establish causation.

## Part B

Part B reads `data/clean/orders.csv`, `data/clean/customers.csv`, and `data/clean/products.csv`. Customer behaviour uses completed orders. The lapse features stop at 2025-09-01. No encoder, scaler, or resampler was fit on test rows. No order dated 2025-09-01 or later entered the features, and the segment labels were not used as features.

### RFM and K-means

The snapshot is 2025-10-01 00:00:00. Recency is the whole days from that snapshot to the customer's latest completed order. Frequency is the number of distinct completed orders. Monetary is the sum of completed `order_amount`. The table has 1,175 customers. Recency has median 19, minimum 0, and maximum 180. Frequency has mean 9.46 and maximum 33. Monetary value has mean INR 2,552.06 and maximum INR 14,766.55.

The three columns were standardised with `StandardScaler`. `KMeans(n_clusters=K, n_init=10, random_state=42)` was fit for every K from 2 to 8.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is the chosen segmentation. It is inside the set {3, 4, 5}, the inertia drop flattens after it, and its silhouette of 0.526 is the best in the table. Inertia at K=8 is 366.5, far below the 1,820.8 at K=2, because every extra centroid can only shrink the within-cluster sum of squares. That decline continues until each customer is their own cluster, so the fall by itself cannot choose K.

Cluster ids are labels with no ordering. Names come from the profile: the highest mean monetary value is Premium, the highest mean frequency among the rest is Frequent, the highest mean recency among the rest is At risk, and the remaining segment is Occasional. `is_premium` is true only for Premium.

| Segment id | Name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional | 436 | 28.86 | 3.77 | 1,057.62 | A small basket-building offer to lift a second order |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | A personal outreach and a high-value retention reward |
| 2 | Frequent | 419 | 8.77 | 19.40 | 3,410.02 | Refill reminders for grocery and dairy baskets |
| 3 | At risk | 182 | 109.04 | 2.20 | 685.99 | A win-back coupon while the gap is still recoverable |

The 138-customer Premium segment has mean monetary value about INR 7,130 and mean frequency about 6.8. The 419-customer Frequent segment has mean frequency about 19.4 and mean recency about 8.8 days.

`PCA(n_components=2)` on the scaled RFM matrix explains 0.634 and 0.218 of the variance, 0.852 in total. `outputs/customer_segments.csv` has columns `customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name`, and `is_premium`.

### Lapse label and split

The cutoff is 2025-09-01 00:00:00. The cohort is every customer with at least one completed order before that cutoff: 1,156 customers. `lapsed` is 1 when the customer has no completed order from 2025-09-01 through 2025-09-30, and 0 otherwise. 443 customers lapsed, 38.32%. The features are recency from the cutoff, frequency, monetary value, average discount, cancellation rate, average delivery minutes, favourite category (ties broken by category name), city, age, gender, and tenure in days. The feature table has no missing values.

`train_test_split(test_size=0.2, stratify=y, random_state=42)` gives 924 training rows and 232 test rows. Always predicting "not lapsed" on the test set has accuracy 61.6% and a lapsed-class F1 of 0.

City, gender, and favourite category are one-hot encoded with `OneHotEncoder(drop="first")` inside a pipeline, and the numeric features are standardised. Both steps are fit on the training rows only. The fitted scaler's mean of frequency is 8.11. The full-cohort mean of frequency is 8.03. They differ because the scaler was fit on the training rows alone and never saw the test rows.

### Tuned models

`LogisticRegression(max_iter=1000)` was searched over `C` in {0.01, 0.1, 1, 10, 100} with `GridSearchCV`, `cv=5`, and `scoring="f1"`. The best setting is `C=100`, with cross-validated F1 0.651. On the untouched test set the lapsed-class precision is 0.644, recall is 0.652, and F1 is 0.648. The confusion matrix, with rows actual not-lapsed then lapsed and columns the same predictions, is 111, 32, 31, 58.

`RandomForestClassifier(n_estimators=100, random_state=42)` was searched over `max_depth` in {3, 5, 8} and `min_samples_leaf` in {5, 10, 20}, the same 5-fold F1 search (45 fits). The best setting is depth 8 and leaf 10, with cross-validated F1 0.681. On the test set the lapsed-class precision is 0.659, recall is 0.652, and F1 is 0.655. The confusion matrix is 113, 30, 31, 58.

The random forest is the selected model, with a lapsed-class test F1 of 0.655 against the logistic model's 0.648 and a baseline F1 of 0. Both tuned models find the same 58 of 89 lapsing test customers, and the forest makes 30 false alarms against the logistic model's 32. Missing a customer who is about to stop costs the relationship, while a false alarm only spends a retention offer on someone who would have stayed. The forest is saved because it keeps that recall and trims a few wasted offers.

### Class balance

The training split has 570 customers who did not lapse and 354 who did. Lapsed training rows were duplicated with `numpy.random.RandomState(42)` until both classes had 570 rows. The best logistic regression, `C=100`, was refit on that balanced training set only and scored on the original test set. Precision moved from 0.644 to 0.589, recall from 0.652 to 0.854, and F1 from 0.648 to 0.697. Recall moved the most. Precision stayed near 0.6 and did not rise. The retention team would reach 76 of the 89 lapsing test customers instead of 58. The extra calls include more customers who kept ordering, so the contact list gets longer.

### What the model uses

The five strongest random-forest importances are frequency 0.280, monetary value 0.271, recency 0.115, average delivery minutes 0.061, and cancellation rate 0.052.

In the unbalanced logistic regression the recency coefficient is +0.040: a longer gap since the last order raises the lapse chance, so a quiet customer should be contacted first. The frequency coefficient is -1.698: customers who order more often are less likely to lapse, so a drop in order count is a warning. The monetary coefficient is +0.285: higher completed spend raises the lapse chance in this fit, so spend alone is not a loyalty signal once order count is held constant.

`outputs/lapse_model.joblib` is the selected forest pipeline. Reloading it with `joblib.load` reproduces the original test predictions exactly.

Among test customers in the Premium segment, the ten highest predicted lapse probabilities are written to `outputs/retention_list.csv` with columns `customer_id`, `lapse_probability`, and `actually_lapsed`. Two of those ten actually lapsed, C0525 and C0565. The highest probability is C0862 at 0.491, and every probability on the list is below 0.50. The Premium flag comes from the whole-window RFM, which includes the September orders that define the lapse label, so this retention list is illustrative and not a leakage-free evaluation.

## Part C

Part C reads `data/raw/tickets.csv` in file order, with `keep_default_na=False`, and routes classified tickets with `outputs/customer_segments.csv`. Every full run calls `mock_llm.reset()` first. The user message sent to the model is the redacted ticket text alone.

### Guardrails and redaction

A ticket whose text is empty or only whitespace is `skipped_empty` and is never sent. There are 4. A ticket whose `est_tokens` estimate is above 400 is `flagged_long` and is never sent. There are 4, and each estimates to 734 tokens. The other 156 tickets are eligible for a model call.

Personal details are replaced, in order, before any prompt is built: email addresses become `[EMAIL]`, phone numbers become `[PHONE]`, and order ids become `[ORDER_ID]`. Across all 164 tickets that is 55 emails, 91 phone numbers, and 132 order ids, 278 substitutions in total. Applying the same three patterns to the redacted texts finds nothing. The script also refuses to call the model if any of those patterns appears in the system prompt or the user message.

### Calls, retries, and accepted answers

Both prompts contain `CLASSIFY`, the four categories `billing`, `delivery`, `technical`, and `fraud`, and a JSON-only instruction of the form `{"category": "...", "reason": "..."}`. The zero-shot prompt does not contain `EXAMPLES:`. The few-shot prompt contains `EXAMPLES:` and two labelled examples, one of them fraud. The examples use the same placeholders, so they do not put an email, a phone number, or an order id into the prompt.

A `TransientLLMError` is retried up to three times. The wait before retry number n is `0.05 × 2^n × random.uniform(0.9, 1.15)` seconds, and each retry logs the ticket id, the retry number, and the wait. A full run raises 19 transient errors. With this mock every one is retry number 0, the waits fall between 0.045 and 0.0575 seconds, and all 19 are recovered. No ticket ends in a permanent error.

A reply is `classified_json` when it parses as JSON and the category is one of the four names: 135 tickets. A reply that is not JSON is `classified_keyword` when the lowercase words contain exactly one of those category names: 12 tickets. Those 12 are the mock's chatty sentences, and the word pattern is what separates `billing-related` into `billing` and `related`. Every other reply, including valid JSON whose category is outside the list, is `flagged_invalid_output`: 9 tickets. Those 9 are the mock's `shipping` replies. They go to a human. The pipeline does not guess a category for them.

### Zero-shot and few-shot

Each comparison run has no token budget. Both classify 147 tickets (135 JSON plus 12 keyword). Zero-shot uses 11,428 tokens, 77.74 per classified ticket, and is correct on 126 of 147, which is 85.7%. Few-shot uses 18,292 tokens, 124.44 per classified ticket, and is correct on all 147, which is 100%.

Zero-shot confusion, counted only on the 147 classified tickets:

| True category | Predicted billing | Predicted delivery | Predicted fraud | Predicted technical |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

Every zero-shot error is a fraud ticket predicted billing (11) or a technical ticket predicted billing (10). Billing and delivery were already exact.

The few-shot prompt costs 6,864 extra tokens on this batch, 46.70 more per classified ticket. Those tokens correct all 21 mistakes, including the 11 fraud tickets that zero-shot would have labelled billing and kept off the security queue. The extra tokens are worth paying when a missed fraud report is more expensive than about 47 tokens per ticket. On this batch the billing and delivery rows were already exact, so the extra cost is buying the fraud and technical corrections, not those two rows.

### Budget and routing

Before each model call, if the tokens already spent are at least the budget, the ticket is `budget_exhausted` and is not sent. The zero-shot demo uses `TOKEN_BUDGET = 6000` and writes `outputs/triage_budget_demo.csv`. It stops 74 tickets.

`outputs/triage_results.csv` is the few-shot run with `TOKEN_BUDGET = 100000`. It has 164 rows and no ticket is stopped by that budget. A customer missing from the segment file would be treated as not premium; every ticket customer in this file has a segment. `category` is blank for the 17 tickets that were not classified. Routing:

| Status or category | Tickets | Route | Priority |
|---|---:|---|---|
| skipped_empty | 4 | human_review | NORMAL |
| flagged_long | 4 | human_review | NORMAL |
| flagged_invalid_output | 9 | human_review | NORMAL |
| fraud, after a rule check | 34 | security_team | HIGH |
| other classified Premium customers | 11 | premium_desk | HIGH |
| other classified customers | 102 | support_queue | NORMAL |

The 135 JSON replies and the 12 keyword replies were accepted by those rules. The 4 empty tickets, the 4 over-long tickets, and the 9 invalid replies were not accepted and went to a human. The final file sends 17 tickets to `human_review`.

## Part D

Part D reads the five files in `data/policies/`, the questions in `data/raw/policy_eval.csv`, the seven requests in `data/raw/agent_scenarios.csv`, and orders from `data/instamart.db`. The refund agent calls the same `retrieve` function as the policy assistant. Both model behaviours come from `mock_llm.py`.

### Policy retrieval

Each document is split into 40-word chunks with a 10-word overlap: words 0–39, then 30–69, and so on, stopping once a chunk reaches the last word. That produces 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2.

A `TfidfVectorizer` is fit on those chunk texts. Term frequency measures how often a word appears in a chunk. Inverse document frequency down-weights a word that appears in many chunks, so a rarer word counts for more. Two phrasings of the same rule do not match when they share no words, which is the limit of this lexical score.

`retrieve("refund policy late delivery", k=3)` returns three chunks, and all three come from `refund_policy.txt`. `answer` sends a system message containing `POLICY ASSISTANT` and a user message that starts with `CONTEXT:`, lists the three chunks with their source file names, and ends with `QUESTION:`.

On the 14 evaluation questions, with the 10-word overlap, all 12 answerable questions have the expected document in the top three sources and the expected fact in the answer (12/12 and 12/12). Both unanswerable questions receive the fallback that begins `I could not find` (2/2).

The same evaluation with overlap 0 still retrieves the right document for all 12, and the fallback still covers both unanswerable questions, but only 10 of 12 answers contain the expected fact. One question that breaks is "What delivery time makes a completed order eligible for a full refund?" The words "more than 45 minutes" end the first 40-word chunk, and the next chunk starts at "receives a full refund of the order amount." The model returns that second fragment, which does not contain the fact. The same boundary cuts the 48-hour payment sentence, so "How soon is money returned when a payment fails but is deducted?" fails too. The 10-word overlap keeps each of those sentences inside a chunk.

### Refund scenarios

The five tools are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. A completed order delivered in more than 45 minutes is refunded in full. A delivery of 45 minutes or less refunds INR 0. An amount above INR 1,000 is escalated and is not issued automatically. Every simulated request is treated as inside the 7-day window. `outputs/agent_runs.jsonl` has one line for each scenario.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 804 |
| S2 | REFUND | 135.00 | 5 | 804 |
| S3 | ESCALATE | 1,999.00 | 5 | 791 |
| S4 | DENIED | 0.00 | 4 | 543 |
| S5 | DENIED | 0.00 | 3 | 305 |
| S6 | NOT_FOUND | 0.00 | 2 | 76 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 2,673 |

S7 finishes all six model calls under the default token limit of 6,000. Its final answer is a short handoff to a human.

The S1 run with `token_limit=200` ends `FALLBACK_TOKEN_LIMIT` after 3 model calls and 322 tokens, without executing the tool requested on the call that crossed the limit. The run with `time_limit_seconds=0` ends `FALLBACK_TIMEOUT` before the first model call.

The crash demonstration saves a checkpoint for run `s1-crash` after the second tool, then `resume_agent` continues that checkpoint. The tools executed across the crash and the resume, in order, are `get_order`, `search_policy`, `calculate_refund`, `issue_refund`. None is repeated. The final status is `REFUND` and the amount is INR 128.25.

`outputs/refund_ledger.jsonl` records the issued refunds for S1 (INR 128.25) and S2 (INR 135.00), the escalation for S3 (INR 1,999.00), and a second issue of INR 128.25 from the crash demonstration. Checkpoints for these runs are in `state/`.

### ReAct and plan-and-execute

This refund agent is a ReAct loop. S1, a late order under INR 1,000, takes 5 model calls and 804 tokens: look up the order, search the policy, calculate, issue, then confirm. S5, a cancelled order, stops after 3 calls and 305 tokens: the lookup returns cancelled, the policy search follows, and the agent denies the refund. S6, an unknown order id, stops after 2 calls and 76 tokens when `get_order` returns not found. S7 asks for the same policy search on every turn, spends 6 calls and 2,673 tokens, and the iteration cap hands the case to a human.

A plan-and-execute design would ask the model once for the whole step list, run those tools, and call the model again only when a step returns an error. S1 would cost about two model calls instead of five. S6 would still need a second call after the missing order, and S5 would need one after the cancelled status. The saving is real on the straight refund. ReAct spends the extra calls so the next tool is chosen after the observation, which is the branch this workflow actually depends on. I would keep ReAct here. The counts stay small, a cancelled order and a missing id are ordinary outcomes, and the six-call cap already stops a loop.

## Limitations and Next Steps

City-median ages replace 67 missing or impossible ages with a typical age for that city, so those 67 values are not observed ages. The 774 imputed delivery times are category medians among completed orders, not the minutes a rider actually took. The 112 repaired amounts trust the catalogue price and the recorded discount. Cancelled orders keep a missing delivery time on purpose, so any average of delivery time must ignore them.

No encoder, scaler, or resampler was fit on the lapse test rows. No order dated 2025-09-01 or later entered the lapse features, and the segment labels were not used as features. The Premium flag is a post-hoc whole-window label because that RFM window includes September, the month that defines the lapse label, so the retention list is illustrative and not a leakage-free evaluation.

Empty tickets, over-long tickets, replies that fail the category rule, and tickets stopped by the token budget go to a human. The triage pipeline does not guess a category for them.

Policy retrieval is lexical. A question that uses different words from the document can miss the sentence that holds the answer, which is why the zero-overlap chunking already drops two facts that the 10-word overlap keeps. The refund agent treats every request as inside the policy's 7-day window, so it does not check the delivery date itself. A refund above INR 1,000 is recorded for a human lead and is not issued by the agent.
