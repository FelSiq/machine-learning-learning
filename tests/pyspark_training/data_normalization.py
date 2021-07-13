import pyspark
import pyspark.sql
import pyspark.sql.types as pst
import pyspark.ml.feature
from pyspark.sql import functions


sc = pyspark.SparkContext("local", "Data Standardization")
ss = pyspark.sql.SparkSession(sc)

schema = pst.StructType(
    [
        pst.StructField("col_A", pst.FloatType()),
        pst.StructField("col_B", pst.FloatType()),
        pst.StructField("col_C", pst.FloatType()),
        pst.StructField("col_D", pst.FloatType()),
        pst.StructField("col_E", pst.FloatType()),
        pst.StructField("col_F", pst.FloatType()),
        pst.StructField("col_G", pst.FloatType()),
    ]
)

df = ss.read.csv("data.csv", schema=schema, header=True)

vector_assembler = pyspark.ml.feature.VectorAssembler(inputCols=df.columns, outputCol="SS_features")
temp_train = vector_assembler.transform(df)

# 1) StandardScaler
scaler = pyspark.ml.feature.StandardScaler(inputCol="SS_features", outputCol="standardized")
res = scaler.fit(temp_train).transform(temp_train)
res.select("standardized").show(5, truncate=False)


# 2) MinMaxScaler
scaler = pyspark.ml.feature.MinMaxScaler(min=0.0, max=1.0, inputCol="SS_features", outputCol="standardized")
res = scaler.fit(temp_train).transform(temp_train)
res.select("standardized").show(5, truncate=False)
