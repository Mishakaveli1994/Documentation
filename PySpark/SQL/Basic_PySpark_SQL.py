'''PySpark SQL is one of the most used PySpark modules which is used for processing structured
columnar data format. Once you have a DataFrame created, you can interact with the data by using SQL syntax.

In other words, Spark SQL brings native RAW SQL queries on Spark meaning you can run traditional
ANSI SQL’s on Spark Dataframe, in the later section of this PySpark SQL tutorial, you will learn
in detail using SQL select, where, group by, join, union e.t.c

In order to use SQL, first, create a temporary table on DataFrame using createOrReplaceTempView()
function. Once created, this table can be accessed throughout the SparkSession_SparkContext using sql() and it will
be dropped along with your SparkContext termination.

Use sql() method of the SparkSession_SparkContext object to run the query and this method returns a new DataFrame.'''

# df.createOrReplaceTempView("PERSON_DATA")
# df2 = spark.sql("SELECT * from PERSON_DATA")
# df2.printSchema()
# df2.show()

'''GroupBy'''
# groupDF = spark.sql("SELECT gender, count(*) from PERSON_DATA group by gender")
# groupDF.show()


# Working with SparkSQL

'''
Using SparkSession_SparkContext you can access PySpark/Spark SQL capabilities in PySpark. 
In order to use SQL features first, you need to create a temporary view in PySpark. 
Once you have a temporary view you can run any ANSI SQL queries using spark.sql() method.

df.createOrReplaceTempView("sample_table")
df2 = spark.sql("SELECT _1,_2 FROM sample_table")
df2.show()

PySpark SQL temporary views are session-scoped and will not be available if the session that creates it 
terminates. If you want to have a temporary view that is shared among all sessions and keep alive until 
the Spark application terminates, you can create a global temporary view using createGlobalTempView()
'''
