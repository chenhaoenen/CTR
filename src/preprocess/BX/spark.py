# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-17 16:46
# Description:  
#--------------------------------------------
import os
from pyspark.sql import SparkSession
# from src.preprocess.BX.utils import parse_arguments
from utils import parse_arguments
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, IntegerType
from pyspark.sql.functions import udf, from_json, col, desc

from pyspark.sql.types import StructType, StructField, LongType, StringType
from pyspark.sql import Row

def setup_spark():
    global sc
    sc = SparkSession.builder.appName("process").getOrCreate()
    sc.sparkContext.setLogLevel("ERROR")
    sc.conf.set('spark.rapids.sql.enabled', 'true')
    sc.conf.set('spark.rapids.sql.explain', 'ALL')
    sc.conf.set('spark.rapids.sql.incompatibleOps.enabled', 'true')
    sc.conf.set('spark.rapids.sql.batchSizeBytes', '512M')
    sc.conf.set('spark.rapids.sql.reader.batchSizeBytes', '768M')


def read_file(args):
    args.users_path = os.path.join(args.input_dir,  'BX-Users.csv')
    args.books_path = os.path.join(args.input_dir,  'BX-Books.csv')
    args.rate_path = os.path.join(args.input_dir, 'BX-Book-Ratings.csv')

    user_schema = StructType(
        [
            StructField('uid', IntegerType(), True),
            StructField('location', StringType(), True),
            StructField('age', IntegerType(), True)
        ]
    )

    book_schema = StructType(
        [
            StructField('bid', IntegerType(), True),
            StructField('title', StringType(), True),
            StructField('author', StringType(), True),
            StructField('year', IntegerType(), True),
            StructField('publisher', StringType(), True)
        ]
    )

    rate_schema = StructType(
        [
            StructField('uid', IntegerType(), True),
            StructField('bid', IntegerType(), True),
            StructField('rate', IntegerType(), True)
        ]
    )

    df_users = sc.read.csv(args.users_path, header=True, schema=user_schema, sep=';')
    df_books = sc.read.csv(args.books_path, header=True, schema=book_schema, sep=';', maxColumns=5)
    df_rate = sc.read.csv(args.rate_path, header=True, schema=rate_schema, sep=';')

    return df_users, df_books, df_rate

def main():
    args = parse_arguments()
    setup_spark()

    df_u, df_i, df_r = read_file(args)
    print('='*100 + args.users_path)
    df_u.show()
    df_u.printSchema()
    df_u.select('Age').summary().show()
    df_u.createOrReplaceTempView("users")
    df2 = sc.sql("select count(*) from (select Age, count(*) from users group by Age) as a")
    df2.show()
    print('='*100 + args.users_path)
    df_i.show()



if __name__ == '__main__':
    main()
