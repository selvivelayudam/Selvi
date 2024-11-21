--SQL Advance Case Study
use db_SQLCaseStudies_advanced
	select * from DIM_CUSTOMER
	select * from DIM_DATE
	select * from DIM_LOCATION
	select * from DIM_MANUFACTURER
	select * from DIM_MODEL
	select * from FACT_TRANSACTIONS
--Q1--BEGIN 
/*1. List all the states in which we have customers who have bought cellphones 
from 2005 till today. */
                                                                                                                           
SELECT [State] FROM DIM_LOCATION A INNER JOIN 
FACT_TRANSACTIONS B ON A.IDLocation = B.IDLocation 
WHERE year(B.[Date])>=2005 group by A.[State];

 

--Q1--END

--Q2--BEGIN

/*2. What state in the US is buying the most 'Samsung' cell phones? */	

select [state] from (select Top 1 L.[State],sum(F.Quantity) as [Tot_qty] from FACT_TRANSACTIONS as F inner join DIM_LOCATION as L on F.IDLocation=L.IDLocation
inner join DIM_MODEL as MO on MO.IDModel=F.IDModel
inner join DIM_MANUFACTURER as M on M.IDManufacturer=MO.IDManufacturer
where M.Manufacturer_Name='Samsung' and L.Country='US'
group by L.[State]
order by [Tot_qty] desc) as T1;


--Q2--END



--Q3--BEGIN  
--3. Show the number of transactions for each model per zip code per state.     
	
	select L.ZipCode ,MO.IDModel,L.[State],count(IDCustomer) as TOT_Transaction from FACT_TRANSACTIONS as F
	inner join DIM_LOCATION as L on F.IDLocation=L.IDLocation
	inner join DIM_MODEL as MO on MO.IDModel=F.IDModel
	group by L.[State], L.ZipCode ,MO.IDModel;


--Q3--END



--Q4--BEGIN
--4. Show the cheapest cellphone (Output should contain the price also)

select Mo.MOdel_NAme,F.Totalprice from FACT_TRANSACTIONS as F
inner join DIM_MODEL as Mo on Mo.IDModel=F.IDModel
order by F.TotalPrice;


--Q4--END

--Q5--BEGIN

/*5. Find out the average price for each model in the top5 manufacturers in 
terms of sales quantity and order by average price.*/

select MANUFACTURER_NAME, MODEL_NAME,AVG(TOTALPRICE) AS AVG_PRICE FROM FACT_TRANSACTIONS T1 LEFT JOIN DIM_MODEL T2 ON 
T1.IDModel = T2.IDModel LEFT JOIN DIM_MANUFACTURER T3 ON T2.IDManufacturer = T3.IDManufacturer
WHERE Manufacturer_Name IN 
(
SELECT TOP 5 Manufacturer_Name FROM FACT_TRANSACTIONS T1 LEFT JOIN DIM_MODEL AS T2 ON T1.IDModel = T2.IDModel
LEFT JOIN DIM_MANUFACTURER T3 ON T2.IDManufacturer = T3.IDManufacturer
GROUP BY Manufacturer_Name
ORDER BY SUM(Quantity) DESC
)
GROUP BY Manufacturer_Name,Model_Name
ORDER BY AVG(TotalPrice) DESC;

--Q5--END

--Q6--BEGIN
/*6. List the names of the customers and the average amount spent in 2009, 
where the average is higher than 500 */


Select C.Customer_Name, AVG(F.TotalPrice) as [Average Amount] from DIM_CUSTOMER as C 
inner join FACT_TRANSACTIONS as F on C.IDCustomer=F.IDCustomer
where YEAR(F.[Date])=2009 
group by C.Customer_Name,C.IDCustomer
having  AVG(F.TotalPrice)>500;



--Q6--END
	
--Q7--BEGIN  
/*7. List if there is any model that was in the top 5 in terms of quantity, 
simultaneously in 2008, 2009 and 2010 */

	
Select T1.Model_Name  from (select top 5 Mo.Model_Name,Mo.IDmodel ,SUM(F.Quantity) as TotQty from FACT_TRANSACTIONS as F
inner join DIM_MODEL as Mo on Mo.IDModel=F.IDModel
where YEAR(F.[Date])=2008 
group by Mo.Model_Name,Mo.IDmodel
order by TotQty desc) as T1 inner join
(select top 5 Mo.Model_Name,Mo.IDmodel ,SUM(F.Quantity) as TotQty from FACT_TRANSACTIONS as F
inner join DIM_MODEL as Mo on Mo.IDModel=F.IDModel
where YEAR(F.[Date])=2009
group by Mo.Model_Name,Mo.IDmodel
order by TotQty desc) as T2 on T1.Model_Name=T2.Model_Name inner join
(select top 5 Mo.Model_Name,Mo.IDmodel ,SUM(F.Quantity) as TotQty from FACT_TRANSACTIONS as F
inner join DIM_MODEL as Mo on Mo.IDModel=F.IDModel
where YEAR(F.[Date])=2010
group by Mo.Model_Name,Mo.IDmodel
order by TotQty desc) as T3 on T3.Model_Name=T2.Model_Name;	





--Q7--END	
--Q8--BEGIN
/*8. Show the manufacturer with the 2nd top sales in the year of 2009 and the 
manufacturer with the 2nd top sales in the year of 2010. */



select * from( select row_number() over(order by sum(F.TotalPrice) desc) as R, M.Manufacturer_Name ,sum(F.TotalPrice) as Tot_Amt,
YEAR(F.[Date]) as [Year]
from FACT_TRANSACTIONS as F 
inner join DIM_MODEL As Mo on Mo.IDModel=F.IDModel
inner join DIM_MANUFACTURER as M on M.IDManufacturer=Mo.IDManufacturer
where YEAR(F.[Date])=2009 
group by  M.Manufacturer_Name,YEAR(F.[Date]))as T1
where R=2 union all
select * from (select row_number() over(order by sum(F.TotalPrice) desc) as R, M.Manufacturer_Name ,sum(F.TotalPrice) as Tot_Amt,
YEAR(F.[Date]) as [Year]
from FACT_TRANSACTIONS as F 
inner join DIM_MODEL As Mo on Mo.IDModel=F.IDModel
inner join DIM_MANUFACTURER as M on M.IDManufacturer=Mo.IDManufacturer
where YEAR(F.[Date])=2010 
group by  M.Manufacturer_Name,YEAR(F.[Date])) as T2
where R=2; 



--Q8--END

--Q9--BEGIN
--9. Show the manufacturers that sold cellphones in 2010 but did not in 2009. 
	

select Manufacturer_Name from (select  M.Manufacturer_Name ,sum(F.TotalPrice) as Tot_Amt
from FACT_TRANSACTIONS as F 
inner join DIM_MODEL As Mo on Mo.IDModel=F.IDModel
inner join DIM_MANUFACTURER as M on M.IDManufacturer=Mo.IDManufacturer
where YEAR(F.[Date])=2010
group by M.Manufacturer_Name , YEAR(F.[Date]))as T1
except
select Manufacturer_Name from (select  M.Manufacturer_Name ,sum(F.TotalPrice) as Tot_Amt
from FACT_TRANSACTIONS as F 
inner join DIM_MODEL As Mo on Mo.IDModel=F.IDModel
inner join DIM_MANUFACTURER as M on M.IDManufacturer=Mo.IDManufacturer
where YEAR(F.[Date])=2009
group by M.Manufacturer_Name , YEAR(F.[Date])) as T2;	



--Q9--END

--Q10--BEGIN
/*10. Find top 100 customers and their average spend, average quantity by each 
year. Also find the percentage of change in their spend.*/
	

select TBL1.IDCustomer,TBL1.Customer_Name , TBL1.[Year],TBL1.Avg_Spend,TBL1.Avg_Qty,case when TBL2.[Year] is not null then
((TBL1.Avg_Spend-TBL2.Avg_Spend)/TBL2.Avg_Spend )* 100 
else NULL
end as 'YOY in Average Spend' from
(select C.IDcustomer,C.Customer_Name,AVG(F.TotalPrice) as Avg_Spend ,AVG(F.Quantity) as Avg_Qty ,
YEAR(F.Date) as [Year] from DIM_CUSTOMER as c 
left join FACT_TRANSACTIONS as F on F.IDCustomer=C.IDCustomer 
where C.IDCustomer in (Select top 10 C.IDCustomer from DIM_CUSTOMER as c 
left join FACT_TRANSACTIONS as F on F.IDCustomer=C.IDCustomer 
group by C.IDCustomer 
order by Sum(F.TotalPrice) desc)
group by C.IDcustomer,C.Customer_Name,YEAR(F.Date)) as TBL1 
left join 
(select C.IDcustomer,C.Customer_Name,AVG(F.TotalPrice) as Avg_Spend ,AVG(F.Quantity) as Avg_Qty ,
YEAR(F.Date) as [Year] from DIM_CUSTOMER as c 
left join FACT_TRANSACTIONS as F on F.IDCustomer=C.IDCustomer 
where C.IDCustomer in (Select top 10 C.IDCustomer from DIM_CUSTOMER as c 
left join FACT_TRANSACTIONS as F on F.IDCustomer=C.IDCustomer 
group by C.IDCustomer 
order by Sum(F.TotalPrice) desc)
group by C.IDcustomer,C.Customer_Name,YEAR(F.Date)) as TBL2 
on TBL1.IDCustomer=TBL2.IDCustomer and TBL2.[Year]=TBL1.[Year]-1;


--Q10--END
	