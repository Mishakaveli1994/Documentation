# Spark Session Builder

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[1]") \
    .appName('SparkByExamples.com') \
    .getOrCreate()

'''
master() – If you are running it on the cluster you need to use your master name as an argument to master(). 
Usually, it would be either yarn or mesos depends on your cluster setup.

Use local[x] when running in Standalone mode. x should be an integer value and should be greater than 0; 
this represents how many partitions it should create when using RDD, DataFrame, and Dataset. Ideally, x 
value should be the number of CPU cores you have.
appName() – Used to set your application name.

getOrCreate() – This returns a SparkSession_SparkContext object if already exists, and creates a new one if not exist.

Note:  SparkSession_SparkContext object spark is by default available in the PySpark shell.
'''

# Create new session with the same app name
'''
This uses the same app name, master as the existing session. Underlying SparkContext will be 
the same for both sessions as you can have only one context per PySpark application.

spark2 = SparkSession_SparkContext.newSession
print(spark2)
'''

# Get existing SparkSession_SparkContext

'''
spark3 = SparkSession_SparkContext.builder.getOrCreate
print(spark3)
'''

# To set configuration, use config

'''
spark = SparkSession_SparkContext.builder \
      .master("local[1]") \
      .appName("SparkByExamples.com") \
      .config("spark.some.config.option", "config-value") \
      .getOrCreate()
'''

# Create SparkSession_SparkContext with Hive Enable

'''
Hive - Apache Hive is a data warehouse software project built on top of Apache Hadoop for providing 
data query and analysis. Hive gives an SQL-like interface to query data stored 
in various databases and file systems that integrate with Hadoop.

spark = SparkSession_SparkContext.builder \
      .master("local[1]") \
      .appName("SparkByExamples.com") \
      .config("spark.sql.warehouse.dir", "<path>/spark-warehouse") \
      .enableHiveSupport() \
      .getOrCreate()
      
'''

# Create Hive Table

'''
Note that in order to do this for testing you don’t need Hive to be installed. 
saveAsTable() creates Hive managed table. Query the table using spark.sql().

# Create Hive table & query it.  
spark.table("sample_table").write.saveAsTable("sample_hive_table")
df3 = spark.sql("SELECT _1,_2 FROM sample_hive_table")
df3.show()
'''

# Using PySpark Configs
'''
# Set Config
spark.conf.set("spark.executor.memory", "5g")

# Get a Spark Config
partions = spark.conf.get("spark.sql.shuffle.partitions")
print(partions)
'''

# Working with Catalogs

'''
To get the catalog metadata, PySpark Session exposes catalog variable. Note that these methods spark.catalog.listDatabases and spark.catalog.listTables and returns the DataSet.


# Get metadata from the Catalog
# List databases
dbs = spark.catalog.listDatabases()
print(dbs)

# Output
#[Database(name='default', description='default database', 
#locationUri='file:/Users/admin/.spyder-py3/spark-warehouse')]

# List Tables
tbls = spark.catalog.listTables()
print(tbls)

# Example Output
# [Table(name='sample_hive_table', database='default', description=None, tableType='MANAGED', 
# isTemporary=False), Table(name='sample_hive_table1', database='default', description=None, 
# tableType='MANAGED', isTemporary=False), Table(name='sample_hive_table121', database='default', 
# description=None, tableType='MANAGED', isTemporary=False), Table(name='sample_table', database=None, 
# description=None, tableType='TEMPORARY', isTemporary=True)]
'''

# Spark Commonly used methods

'''
version() – Returns the Spark version where your application is running, probably the 
Spark version your cluster is configured with.

createDataFrame() – This creates a DataFrame from a collection and an RDD

getActiveSession() – returns an active Spark session.

read() – Returns an instance of DataFrameReader class, this is used to read records from csv, 
parquet, avro, and more file formats into DataFrame.

readStream() – Returns an instance of DataStreamReader class, this is used to read streaming data. 
That can be used to read streaming data into DataFrame.

sparkContext() – Returns a SparkContext.

sql() – Returns a DataFrame after executing the SQL mentioned.

sqlContext() – Returns SQLContext.

stop() – Stop the current SparkContext.

table() – Returns a DataFrame of a table or view.

udf() – Creates a PySpark UDF to use it on DataFrame, Dataset, and SQL.
'''
