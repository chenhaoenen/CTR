# -*- coding: utf-8 -*-      
#--------------------------------------------
# Author:chenhao
# Date:2020-12-17 16:46
# Description:  
#--------------------------------------------
import os
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from utils import parse_arguments
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import udf, countDistinct, col

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
    schema = StructType(
        [
            StructField('id', StringType(), True),
            StructField('click', IntegerType(), True),
            StructField('hour', IntegerType(), True),
            StructField('c1', StringType(), True),
            StructField('banner_pos', StringType(), True),
            StructField('site_id', StringType(), True),
            StructField('site_domain', StringType(), True),
            StructField('site_category', StringType(), True),
            StructField('app_id', StringType(), True),
            StructField('app_domain', StringType(), True),
            StructField('app_category', StringType(), True),
            StructField('device_id', StringType(), True),
            StructField('device_ip', StringType(), True),
            StructField('device_model', StringType(), True),
            StructField('device_type', StringType(), True),
            StructField('device_conn_type', StringType(), True),
            StructField('c14', StringType(), True),
            StructField('c15', StringType(), True),
            StructField('c16', StringType(), True),
            StructField('c17', StringType(), True),
            StructField('c18', StringType(), True),
            StructField('c19', StringType(), True),
            StructField('c20', StringType(), True),
            StructField('c21', StringType(), True)
        ]
    )
    df = sc.read.csv(args.input_path, header=True, schema=schema, sep=',')

    return df

def main():
    args = parse_arguments()
    setup_spark()
    df = read_file(args)

    # transform
    assembler = VectorAssembler(inputCols=["hour"], outputCol="hour_vector")
    df = assembler.transform(df)
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_idx").fit(df)
                for column in list(set(df.columns) - set(['id',
                                                          'device_ip',
                                                          'hour',
                                                          'click',
                                                          'hour_vector',
                                                          'device_id',
                                                          'device_model',
                                                          'site_domain',
                                                          'site_id',
                                                          'app_id',
                                                          'c14',
                                                          'app_domain',
                                                          'c17',
                                                          'c20']))] \
               + [MinMaxScaler(inputCol='hour_vector', outputCol='hour_scalar').fit(df)]
    pipeline = Pipeline(stages=indexers)
    df = pipeline.fit(df).transform(df)

    func = udf(lambda v: float(v[0]), FloatType())

    df = df.withColumn('hour_std', func('hour_scalar'))
    df = df[[w for w in list(df.columns) if 'idx' in w] + ['hour_std', 'click']].cache()

    # to pandas and make config
    config_pd = df.agg(*(countDistinct(col(c)).alias(c) for c in df.columns)).toPandas()
    multi_index = []
    for c in config_pd.columns:
        if '_idx' in c:
            multi_index.append('sparse')
        elif c == 'click':
            multi_index.append('label')
        else:
            multi_index.append('dense')
    config_pd.columns = pd.MultiIndex.from_tuples(zip(multi_index, config_pd.columns))
    s = config_pd.iloc[0]
    dic = {l: s.xs(l).to_dict() for l in s.index.levels[0]}

    if not os.path.exists(args.output_path):
        os.system('mkdir {}'.format(args.output_path))
    with open(os.path.join(args.output_path, 'config.yaml'), 'w', encoding="utf-8") as fw:
        yaml.dump(dic, fw, default_flow_style=False, indent=4)


    # stats count
    total_num = df.count()
    pos_num = df.filter(df.click == 1).count()
    neg_num = df.filter(df.click != 1).count()
    print('#'*20)
    print('raw totle_num:{} pos_num:{} neg_num:{}'.format(total_num,
                                                          pos_num,
                                                          neg_num))

    # sample
    pos_df = df[df.click == 1]
    neg_df = df[df.click != 1].sample(False, 0.5, seed=1234)
    df = pos_df.union(neg_df)
    print('union totle_num:{} pos_num:{} neg_num:{}'.format(df.count(),
                                                            pos_df.count(),
                                                            neg_df.count()))
    print('#'*20)
    # split dataset
    train_df, val_df =df.randomSplit([0.9, 0.1])
    train_df.repartition(1).write.json(os.path.join(args.output_path, 'train'))
    val_df.repartition(1).write.json(os.path.join(args.output_path, 'val'))

if __name__ == '__main__':
    main()
