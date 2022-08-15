'''
PySpark Window functions are used to calculate results such as the rank, row number e.t.c over a range of input rows.
In this article, I’ve explained the concept of window functions, syntax, and finally how to use them with PySpark
SQL and PySpark DataFrame API. These come in handy when we need to make aggregate operations in a specific window
frame on DataFrame columns.

When possible try to leverage standard library as they are little bit
more compile-time safe, handles null and perform better when compared to UDF’s.
If your application is critical on performance try to avoid using custom UDF at all costs as
these are not great on performance.
'''

# Window Functions

'''
PySpark Window functions operate on a group of rows (like frame, partition) and return a single value 
for every input row. PySpark SQL supports three kinds of window functions:
1. Ranking functions
2. Analytic functions
3. Aggregate functions

The below table defines Ranking and Analytic functions and for aggregate functions, we can use any existing 
aggregate functions as a window function.

To perform an operation on a group first, we need to partition the data using Window.partitionBy() , and for row 
number and rank function we need to additionally order by on partition data using orderBy clause.


row_number(): Column	Returns a sequential number starting from 1 within a window partition
rank(): Column	        Returns the rank of rows within a window partition, with gaps.
percent_rank(): Column	Returns the percentile rank of rows within a window partition.
dense_rank(): Column	Returns the rank of rows within a window partition without any gaps. Where as Rank() returns rank with gaps.
ntile(n: Int): Column	Returns the ntile id in a window partition

cume_dist(): Column   	Returns the cumulative distribution of values within a window partition

lag(e: Column, offset: Int): Column
lag(columnName: String, offset: Int): Column
lag(columnName: String, offset: Int, defaultValue: Any): Column	returns the value that is `offset` rows before the
                                                         current row, and `null` if there is less than `offset` 
                                                         rows before the current row.

lead(columnName: String, offset: Int): Column
lead(columnName: String, offset: Int): Column
lead(columnName: String, offset: Int, defaultValue: Any): Column	returns the value that is `offset` rows after the 
                                                                    current row, and `null` if there is less than 
                                                                    `offset` rows after the current row.
'''
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, rank, dense_rank, percent_rank, ntile, cume_dist, lag, lead

spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()

simpleData = (("James", "Sales", 3000),
              ("Michael", "Sales", 4600),
              ("Robert", "Sales", 4100),
              ("Maria", "Finance", 3000),
              ("James", "Sales", 3000),
              ("Scott", "Finance", 3300),
              ("Jen", "Finance", 3900),
              ("Jeff", "Marketing", 3000),
              ("Kumar", "Marketing", 2000),
              ("Saif", "Sales", 4100))

columns = ["employee_name", "department", "salary"]
df = spark.createDataFrame(data=simpleData, schema=columns)
df.printSchema()
df.show(truncate=False)

# Pyspark Ranking Functions

'''
row_number() window function is used to give the sequential row number starting from 1 to the result of each 
window partition.
'''

print('Row Number')
windowSpec = Window.partitionBy("department").orderBy("salary")

df.withColumn("row_number", row_number().over(windowSpec)).show(truncate=False)

'''
rank() window function is used to provide a rank to the result within a window partition. This function leaves gaps in rank when there are ties.
'''
print("Rank")

df.withColumn("rank", rank().over(windowSpec)).show()

'''
dense_rank() window function is used to get the result with rank of rows within a window partition without any gaps. 
This is similar to rank() function difference being rank function leaves gaps in rank when there are ties.
'''
print("Dense Rank")
df.withColumn("dense_rank", dense_rank().over(windowSpec)).show()

'''
Returns the rank of a value in a data set as a percentage of the data set. This function can be used to evaluate 
the relative standing of a value within a data set. For example, you can use PERCENTRANK to evaluate the 
standing of an aptitude test score among all scores for the test.
'''
print("Percent Rank")
df.withColumn("percent_rank", percent_rank().over(windowSpec)).show()

'''
ntile() window function returns the relative rank of result rows within a window partition. In below example we have 
used 2 as an argument to ntile hence it returns ranking between 2 values (1 and 2)
'''
print("ntile")
df.withColumn("ntile", ntile(2).over(windowSpec)).show()

# Pyspark Window Analytic Functions

'''
cume_dist() window function is used to get the cumulative distribution of values within a window partition.
'''
print("cume_dist")
df.withColumn("cume_dist", cume_dist().over(windowSpec)).show()

'''
We use a Lag() function to access previous rows data as per defined offset value. 
It works similar to a Lead function. In the lead function, we access subsequent rows, but in lag function, 
we access previous rows. It is a useful function in comparing the current row value from the previous row value.
'''
print("Lag")
df.withColumn("lag", lag("salary", 2).over(windowSpec)).show()

'''
In the lead function, we access subsequent rows
'''
df.withColumn("lead", lead("salary", 2).over(windowSpec)).show()

# Aggregate Functions
windowSpecAgg = Window.partitionBy("department")
from pyspark.sql.functions import col, avg, sum, min, max, row_number

df.withColumn("row", row_number().over(windowSpec)) \
    .withColumn("avg", avg(col("salary")).over(windowSpecAgg)) \
    .withColumn("sum", sum(col("salary")).over(windowSpecAgg)) \
    .withColumn("min", min(col("salary")).over(windowSpecAgg)) \
    .withColumn("max", max(col("salary")).over(windowSpecAgg)) \
    .where(col("row") == 1).select("department", "avg", "sum", "min", "max") \
    .show()
