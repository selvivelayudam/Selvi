CREATE DATABASE RETAIL_DB
USE RETAIL_DB 
SELECT * FROM Customer
SELECT * FROM prod_cat_info
SELECT * FROM Transactions


///////////////////////*****DATA PREPARATION AND UNDERSTANDING*****////////////////
--Q1.
SELECT
*
FROM
(SELECT COUNT(customer_Id) AS ROWS_IN_CUST FROM Customer) AS T1,
(SELECT COUNT(prod_cat_code) AS ROWS_IN_PROD FROM prod_cat_info) AS T2,
(SELECT COUNT(transaction_id) AS ROWS_IN_TRANS FROM Transactions) AS T3

--Q2.
SELECT
COUNT(Qty) AS NO_OF_RETURN
FROM
Transactions
WHERE
Qty < 0

--Q3.
UPDATE Transactions
SET tran_date = CONVERT(date,tran_date,103)

--Q4.
SELECT
DATEDIFF(DAY,'2011-02-22','2014-02-28') AS NO_OF_DAYS,
DATEDIFF(MONTH,'2011-02-22','2014-02-28') AS NO_OF_MONTHS,
DATEDIFF(YEAR,'2011-02-22','2014-02-28') AS NO_OF_YEARS

--Q5.
SELECT
prod_cat
FROM
prod_cat_info
WHERE
prod_subcat = 'DIY'



/*****DATA ANALYSIS*****/
--Q1.
SELECT TOP 1
Store_type AS TOP_CHANEL,
COUNT(transaction_id) AS CNT_OF_TRANSACTIONS
FROM
Transactions
GROUP BY
Store_type
ORDER BY
COUNT(transaction_id) DESC

--Q2.
SELECT
Gender,
COUNT(customer_Id) AS CNT_OF_GENDER
FROM
Customer
WHERE
GENDER = 'M' OR GENDER = 'F'
GROUP BY
GENDER

--Q3.
SELECT TOP 1
city_code,
COUNT(customer_Id) AS CNT_OF_CUSTOMERS
FROM
Customer
GROUP BY
city_code
ORDER BY
COUNT(customer_Id) DESC

--Q4.
SELECT
prod_cat,
COUNT(prod_subcat) AS NO_OF_SUB_CATEGORIES
FROM
prod_cat_info
WHERE
prod_cat = 'Books'
GROUP BY
prod_cat

--Q5.
SELECT
MAX(Qty) AS MAX_QTY_ORDERED
FROM
Transactions

--Q6.
SELECT
ROUND(SUM(CAST(total_amt AS FLOAT)),2) AS TOTAL_REVENUE
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
WHERE
prod_cat = 'Electronics' OR prod_cat = 'Books'


--Q7.
SELECT
COUNT(cust_id) AS CNT_OF_CUSTOMERS
FROM
(SELECT
cust_id,
COUNT(transaction_id) AS CN_OF_TRANSACTIONS
FROM
Transactions
WHERE
Qty > 0
GROUP BY
cust_id
HAVING
COUNT(transaction_id) > 10) AS T1

--Q8.
SELECT
ROUND(SUM(CAST(total_amt AS FLOAT)),2) AS TOTAL_REVENUE
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
WHERE
(prod_cat = 'Electronics' OR prod_cat = 'Clothing') AND Store_type = 'Flagship store'

--Q9.
SELECT 
prod_subcat,
SUM(CAST(total_amt AS FLOAT)) AS TOT_REVENUE
FROM
Transactions AS T1
INNER JOIN Customer ON cust_id = customer_Id
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
WHERE
Gender = 'M' AND prod_cat = 'Electronics'
GROUP BY
prod_subcat


--Q10.
SELECT TOP 5
prod_subcat,
total_amt,
CONCAT((CAST(total_amt AS FLOAT)*100) / (SELECT SUM(CAST(total_amt AS FLOAT)) FROM Transactions),'%') AS PERCENTAGE
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
GROUP BY
prod_subcat,
total_amt
ORDER BY
CAST(total_amt AS FLOAT) DESC

--Q11.
SELECT
ROUND(SUM(CAST(total_amt AS FLOAT)),2) AS TOTAL_REVENUE
FROM
(SELECT
total_amt,
DATEDIFF(YEAR,CONVERT(date,DOB,103),tran_date) AS CNT_OF_AGE,
DATEDIFF(DAY,tran_date,(SELECT MAX(tran_date) FROM Transactions)) AS CNT_OF_TRAN_DAYS
FROM
Transactions 
INNER JOIN Customer ON customer_Id = cust_id
WHERE
DATEDIFF(DAY,tran_date,(SELECT MAX(tran_date) FROM Transactions)) <= 30 AND 
(DATEDIFF(YEAR,CONVERT(date,DOB,103),tran_date) >= 25 AND DATEDIFF(YEAR,CONVERT(date,DOB,103),tran_date) <= 35)) AS T1

--Q12.
SELECT TOP 1
prod_cat,
SUM(CAST(Qty AS FLOAT)) AS MAX_RETURNS
FROM
(SELECT
transaction_id,
Qty,
tran_date,
prod_cat,
DATEDIFF(MONTH,tran_date,(SELECT MAX(tran_date) FROM Transactions)) AS LAST_3_MONTH_TRAN
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
WHERE
Qty < 0 AND DATEDIFF(MONTH,tran_date,(SELECT MAX(tran_date) FROM Transactions)) <= 3) AS T3
GROUP BY 
prod_cat
ORDER BY
SUM(CAST(Qty AS FLOAT)) 

--Q13.
SELECT TOP 1
Store_type,
QTY_SOLD,
SALES_AMT
FROM
(SELECT
Store_type,
SUM(CAST(Qty AS FLOAT)) AS QTY_SOLD,
ROUND(SUM(CAST(total_amt AS FLOAT)),2) AS SALES_AMT
FROM
Transactions
GROUP BY
Store_type
) AS T1
ORDER BY
QTY_SOLD DESC


--Q14.
SELECT
prod_cat AS CATEGORIES_ABOVE_AVG
FROM
(SELECT
prod_cat,
ROUND(AVG((CAST(total_amt AS FLOAT))),2) AS AVG_REVENUE
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
GROUP BY
prod_cat) AS T3
GROUP BY
prod_cat
AVG_REVENUE
HAVING
AVG_REVENUE >= (SELECT ROUND(AVG((CAST(total_amt AS FLOAT))),2) FROM Transactions)


--Q15.
SELECT TOP 5
CATEGORY,
QTY_SOLD
FROM
(SELECT
prod_cat AS CATEGORY,
prod_subcat AS SUB_CATEGORY,
ROUND(AVG((CAST(total_amt AS FLOAT))),2) AS AVG_REVENUE,
ROUND(SUM((CAST(total_amt AS FLOAT))),2) AS TOTAL_REVENUE,
SUM(CAST(Qty AS FLOAT)) AS QTY_SOLD
FROM
Transactions AS T1
INNER JOIN prod_cat_info AS T2 ON T2.prod_cat_code = T1.prod_cat_code AND prod_sub_cat_code = prod_subcat_code
GROUP BY
prod_cat,
prod_subcat) AS T3
ORDER BY
QTY_SOLD DESC

