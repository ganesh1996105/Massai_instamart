-- Q1
SELECT customer_id, SUM(order_amount) AS total_large_orders
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2
SELECT c.city, p.category, SUM(o.order_amount) AS revenue
FROM orders AS o
JOIN customers AS c ON o.customer_id = c.customer_id
JOIN products AS p ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3
SELECT product_id, product_name, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC;

-- Q4
SELECT c.customer_id
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5
WITH customer_spend AS (
    SELECT customer_id, SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
avg_spend AS (
    SELECT AVG(total_spend) AS avg_total
    FROM customer_spend
)
SELECT cs.customer_id, cs.total_spend, a.avg_total
FROM customer_spend AS cs
CROSS JOIN avg_spend AS a
WHERE cs.total_spend > a.avg_total;

-- Q6
WITH customer_spend AS (
    SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
    FROM orders AS o
    JOIN customers AS c ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rank_pos,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank_pos,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM customer_spend;

-- Q7
WITH daily AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day, SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (ORDER BY day) AS cumulative_revenue
FROM daily
ORDER BY day;
