-- The schema named nearshoring_logistic is created and used
CREATE DATABASE nearshoring_logistic;
USE nearshoring_logistic;
-- Dimension table for customer data
CREATE TABLE dim_customers (
	customer_id INT PRIMARY KEY,
    customer_segment VARCHAR(50),
    customer_city VARCHAR(100),
    customer_state VARCHAR(100),
    customer_country VARCHAR(50),
    customer_zipcode VARCHAR(20)
);
-- Dimension table for product data
CREATE TABLE dim_products (
	product_card_id INT PRIMARY KEY,
    product_category_id INT,
    category_name VARCHAR(100),
    product_name VARCHAR(255),
    product_price DECIMAL(10,2)
);
-- Dimension table for departments data
CREATE TABLE dim_departments (
	department_id INT PRIMARY KEY,
    department_name VARCHAR(100)
);
-- Fact table for order information, including data from customers, 
-- products and departments
CREATE TABLE fact_orders (
	order_id INT,
	order_item_id INT,
    customer_id INT,
    product_card_id INT,
    department_id INT,
    shipping_mode varchar(50),
    order_date_dateorders DATETIME,
    shipping_date_dateorders DATETIME,
    days_for_shipping_real INT,
    days_for_shipment_scheduled INT,
	order_item_total DECIMAL(10,2),
    order_profit_capped DECIMAL(10,2),
    order_year INT,
    order_month INT,
    order_day_of_week INT,
    is_weekend INT,
    item_quantity INT,
    profit_margin DECIMAL(10,2),
    delivery_variance_days INT,
    -- Assing the Primary Keys for the table
    PRIMARY KEY (order_id, order_item_id),
    -- Identify the Foreign Keys from other tables
    FOREIGN KEY (customer_id) REFERENCES dim_customers(customer_id),
	FOREIGN KEY (product_card_id) REFERENCES dim_products(product_card_id),
    FOREIGN KEY (department_id) REFERENCES dim_departments(department_id)
);

-- Upload data into tables
LOAD DATA LOCAL INFILE '/Users/mariodom/Desktop/Portafolio/Nearshoring/dim_customers.csv'
INTO TABLE dim_customers
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/Users/mariodom/Desktop/Portafolio/Nearshoring/dim_departments.csv'
INTO TABLE dim_departments
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/Users/mariodom/Desktop/Portafolio/Nearshoring/dim_products.csv'
INTO TABLE dim_products
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

LOAD DATA LOCAL INFILE '/Users/mariodom/Desktop/Portafolio/Nearshoring/fact_orders.csv'
INTO TABLE fact_orders
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n'
IGNORE 1 ROWS;

-- Inspect all the tables
SELECT * FROM dim_customers;
SELECT * FROM dim_products;
SELECT * FROM dim_departments;
SELECT * FROM fact_orders;

-- Create indexes for the Foreign keys 
-- Create indexes on the Foreign Keys in the Fact Table to optimize JOIN performance
CREATE INDEX idx_customer ON fact_orders(customer_id);
CREATE INDEX idx_product ON fact_orders(product_card_id);
CREATE INDEX idx_department ON fact_orders(department_id);

-- Generating a View of the Data Warehouse to train the model
CREATE VIEW ns_logistics_ml_training AS
SELECT
    f.shipping_mode,
    c.customer_segment,
    c.customer_city,
    c.customer_state,
    p.category_name,
    p.product_price,
    d.department_name,
    f.days_for_shipment_scheduled,
    f.delivery_variance_days,
    f.order_item_total,
    f.order_profit_capped
FROM  fact_orders as f
INNER JOIN dim_customers as c
	ON f.customer_id = c.customer_id
INNER JOIN dim_products as p
	ON f.product_card_id = p.product_card_id
INNER JOIN dim_departments as d
	ON f.department_id = d.department_id
WHERE 
    f.days_for_shipping_real >= 0
    AND c.customer_state IS NOT NULL;
    
-- Generating a View of the Data Warehouse to train the model
-- with added features to try to improve the model
CREATE VIEW ns_logistics_ml_training_temporal AS
SELECT
    f.shipping_mode,
    c.customer_segment,
    c.customer_city,
    c.customer_state,
    p.category_name,
    p.product_price,
    d.department_name,
    f.days_for_shipment_scheduled,
    f.delivery_variance_days,
    f.order_item_total,
    f.order_profit_capped,
    f.order_month,
    f.order_day_of_week,
    f.is_weekend,
    f.item_quantity,
    f.profit_margin
FROM  fact_orders as f
INNER JOIN dim_customers as c
	ON f.customer_id = c.customer_id
INNER JOIN dim_products as p
	ON f.product_card_id = p.product_card_id
INNER JOIN dim_departments as d
	ON f.department_id = d.department_id
WHERE 
    f.days_for_shipping_real >= 0
    AND c.customer_state IS NOT NULL;
    
-- Generating a View of the Data Warehouse for visualization
CREATE VIEW ns_logistics_visual AS
SELECT
	f.order_id,
    f.shipping_mode,
    f.order_date_dateorders,
    f.shipping_date_dateorders,
    f.days_for_shipment_scheduled,
    f.days_for_shipping_real,
    f.delivery_variance_days,
    c.customer_segment,
    c.customer_city,
    c.customer_state,
    p.category_name,
    p.product_price,
    d.department_name,
    f.order_item_total,
    f.order_profit_capped,
    f.order_year,
    f.order_month,
    f.order_day_of_week,
    f.is_weekend,
    f.item_quantity,
    f.profit_margin
FROM  fact_orders as f
INNER JOIN dim_customers as c
	ON f.customer_id = c.customer_id
INNER JOIN dim_products as p
	ON f.product_card_id = p.product_card_id
INNER JOIN dim_departments as d
	ON f.department_id = d.department_id
WHERE 
    f.days_for_shipping_real >= 0
    AND c.customer_state IS NOT NULL;
    
-- Inspect all the views
SELECT * FROM ns_logistics_ml_training;
SELECT * FROM ns_logistics_ml_training_temporal;
SELECT * FROM ns_logistics_visual;

-- Verify the geographical distribution and average delays across States
SELECT 
    c.customer_state,
    COUNT(f.order_id) AS total_orders,
    AVG(f.delivery_variance_days) AS avg_delay_days,
    AVG((f.delivery_variance_days)*24) as avg_delay_hours,
    SUM(f.order_profit_capped) AS total_profit
FROM fact_orders f
INNER JOIN dim_customers c ON f.customer_id = c.customer_id
GROUP BY c.customer_state
ORDER BY total_orders DESC;

-- Checking for extreme values
SELECT 
    MIN(delivery_variance_days) AS max_early_arrival_days,
    MAX(delivery_variance_days) AS max_late_delay_days,
    AVG(delivery_variance_days) AS mean_variance,
    STDDEV(delivery_variance_days) AS standard_deviation
FROM ns_logistics_visual;
-- From the Standard deviation observed of 1.524, means that 68% of the shipments
-- have a delay.
