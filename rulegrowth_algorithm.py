# Databricks notebook source
"""
This algorithm finds rules for when one condition (the occurrence of itemset I) 
is followed by another condition (the occurrence of itemset J), 
meeting certain frequency criteria. 
For initial valid rules, consisting of one item in itemset I and one item in itemset J, 
it will then search all possible ways to expand the left and right sides of the rules 
while still meeting criteria. 
In other words, all items in itemset I occur before all items in itemset J. 
The order of conditions within itemset I or within itemset J do not matter.

Author:
John P. Powers
This is a PySpark implementation of the RuleGrowth algorithm originally developed by Philippe Fournier-Viger 
and colleagues.

Inputs:
A PySpark DataFrame of sequence data.
Sequences are ordered collections of items.
See "User-defined inputs and value" below for more details.

Outputs:
All valid rules are logged in a PySpark DataFrame "rules" 
which includes itemset I, 
itemset J, 
the support of the rule, 
and the confidence of the rule.

Copyright and licensing:
© 2024, The University of North Carolina at Chapel Hill. Permission is granted to use in accordance with the GNU GPL version 3 license. 

This is an implementation of the RuleGrowth algorithm.

The original RuleGrowth algorithm code is part of the SPMF DATA MINING SOFTWARE copyright by Philippe Fournier-Viger and some parts are copyright by contributors. 
© 2008-2013 Philippe Fournier-Viger
(http://www.philippe-fournier-viger.com/spmf)

The code is licensed under the open-source GNU GPL version 3 license. The GPL license provides four freedoms:
Obtain and run the program for any purpose
Get a copy of the source code
Modify the source code
Re-distribute the modified source code

The only restriction is that if you want to redistribute the source code, you must:
Provide access to the source code
License derived work under the same GPL v3 license

"""

# COMMAND ----------

import pandas as pd
import numpy as np

from pyspark.sql.types import StructType, StructField, StringType, FloatType
import pyspark.sql.functions as F
from pyspark.sql.window import Window

# COMMAND ----------

## User-defined inputs and values

# minimum gap in days between subsequent items to define distinct timepoints/itemsets
# this setting allows items in close temporal proximity to be considered part of the same timepoint
# new timepoint defined after a gap of at least [timepoint_break] days since the 
# previous item in the sequence
timepoint_break = 1

# min_supp is decimal representing the minimum support threshold
# Support of a rule is the proportion of sequences in which the rule occurs
min_supp = 0.02

# min_conf is decimal representing the minimum confidence threshold
# Confidence of a rule: 
# Given that the left side of the rule / antecedent / itemset I occurred, 
# the probability that the right side of the rule / consequent / itemset J will follow
# In other words, the proportion of sequences in which itemset J follows itemset I
# out of the total number of sequences in which itemset I occurs 
min_conf = 0.02

# max number of items that can be included in the left and right side when building potential rules
antecedent_max = 3
consequent_max = 3

# input data are expected to be formatted as a PySpark DataFrame with 3 columns in the following order:
# the sequence identifier (e.g., a patient identifier)
# the date associated with the item (e.g., date of a diagnosis)
# the item identifier (e.g., a diagnosis code)
input_data = your_df

# COMMAND ----------

## Transform dated records to sequences for sequential analysis

# Renaming columns
raw_data = input_data.toDF('seq', 'date', 'item')

# Eliminate multiple instances of the same item on the same date in the same sequence
raw_data = raw_data.distinct()

# Replace dates with sequential timepoint labels to define itemsets
# E.g., items from the first timepoint in a sequence get label 1
# From the second timepoint in a sequence get label 2, etc.
# Distinct timepoints defined based on user-defined timepoint_break
window = Window.partitionBy('seq').orderBy('date')
sequence_data = raw_data.withColumn('delta', F.datediff(F.col('date'), F.lag(F.col('date'), 1).over(window)))
sequence_data = sequence_data.fillna(timepoint_break, 'delta')

sequence_data = sequence_data.withColumn('delta_bin', F.when(sequence_data['delta'] >= timepoint_break, 1).otherwise(0))

window = Window.partitionBy('seq').orderBy('date').rangeBetween(Window.unboundedPreceding, 0)
sequence_data = sequence_data.withColumn('timepoint', F.sum('delta_bin').over(window))

# Eliminate multiple instances of the same item at the same timepoint in the same sequence
sequence_data = sequence_data.select('seq', 'timepoint', 'item')
sequence_data = sequence_data.distinct()

# COMMAND ----------

# For each item for each sequence, 
# create columns storing the first and last timepoints at which it occured
data_first = sequence_data.groupBy(['seq', 'item']).agg(F.min('timepoint').alias('first'))
data_last = sequence_data.groupBy(['seq', 'item']).agg(F.max('timepoint').alias('last'))
item_occurrences = data_first.join(data_last, ['seq', 'item'], 'inner')

# COMMAND ----------

## Algorithm setup

# "rules" will store final rules that pass the user-defined thresholds
columns = StructType([StructField('itemsetA', StringType(), True),
                    StructField('itemsetB', StringType(), True),
                    StructField('support', FloatType(), True),
                    StructField('confidence', FloatType(), True)]) 
rules = spark.createDataFrame(data=[], schema=columns)

# Get total number of sequences
# and convert the decimal support threshold into a count threshold
seq_total = sequence_data.select(F.countDistinct('seq')).collect()[0][0]
min_supp_ct = seq_total * min_supp

# Get frequent items
# i.e., items that occur in at least min_supp_ct sequences
items_by_seq = (
    sequence_data
    .dropDuplicates(['seq', 'item'])
    .select('seq', 'item')
)
item_counts = (
    items_by_seq
    .groupBy('item')
    .agg(F.count('seq').alias('count'))
)
freq_items = (
    item_counts
    .where(item_counts['count'] >= min_supp_ct)
    .select('item')
)

# Reduce item_occurrences to frequent items
item_occurrences = item_occurrences[
    item_occurrences['item']
    .isin(freq_items.rdd.map(lambda x: x['item']).collect())
]

# COMMAND ----------

def expand_right(r_candidates, rules):
    """Expand right side of the rules by 1 item at a time.

    Arguments:
    r_candidates -- dataframe of base rules to try expanding
    rules -- dataframe storing rules for final output

    Returns:
    rules -- updated rules dataframe
    """

    for k in range(1, consequent_max): 

        # Try all frequent items for each rule that are greater than all items in itemset_j
        # This condition is needed to avoid rule duplication in other expansions
        # Add this candidate item (item c) as a new column 
        r_candidates = r_candidates.crossJoin(freq_items.withColumnRenamed('item', 'item_c'))
        r_candidates = r_candidates.where(F.array_max(r_candidates['itemset_j']) < r_candidates['item_c'])

        # Build dataframe of all sequences that contain itemset_i:
        # "candidates_i" (for computing confidence later)
        # and dataframe of all sequences that contain the full candidate rule: 
        # "r_candidates" (for computing support)
        # first_max stores the latest of the first timepoints for each item in itemset_i
        candidates_i = (
            r_candidates
            .join(item_occurrences, 
                r_candidates['itemset_i'][0] == item_occurrences['item'],
                how='inner')
            .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first')
            .withColumnRenamed('first', 'first_max')
        )
        for j in range(1,i):
            candidates_i = (
                candidates_i
                .join(right, 
                    [
                        candidates_i['itemset_i'][j] == right['item'],
                        candidates_i['seq'] == right['seq_r']
                    ],
                    how='inner')
                .withColumn('first_max', F.greatest('first_max', 'first'))
                .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first_max')
            )

        # last_min stores the earliest of the last timepoints for each item in itemset_j
        # plus the candidate item_c
        r_candidates = (
            candidates_i
            .join(right, 
                [
                    candidates_i['item_c'] == right['item'], 
                    candidates_i['seq'] == right['seq_r']
                ], 
                how='inner')
            .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first_max', 'last')
            .withColumnRenamed('last', 'last_min')
        )
        for j in range(i):
            r_candidates = (
                r_candidates
                .join(right, 
                    [
                        r_candidates['itemset_j'][j] == right['item'],
                        r_candidates['seq'] == right['seq_r']
                    ],
                    how='inner')
                .withColumn('last_min', F.least('last_min', 'last'))
                .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first_max', 'last_min')
            )
        
        # first_max < last_min ensures that all items on the right occur after all items on the left
        r_candidates = (
            r_candidates
            .where(F.col('first_max') < F.col('last_min'))
            .select('itemset_i', 'item_c', 'itemset_j', 'seq')
        )
        
        # Check candidates against support threshold
        r_candidates = (
            r_candidates
            .groupBy(['itemset_i', 'item_c', 'itemset_j'])
            .agg(F.count('seq').alias('count'))
            .where(F.col('count') >= min_supp_ct)
        )

        # Compute confidence for candidates and check against confidence threshold
        i_counts = (
            candidates_i
            .select('itemset_i', 'item_c', 'itemset_j', 'seq')
            .groupBy('itemset_i', 'item_c', 'itemset_j')
            .agg(F.count('seq').alias('count_i'))
        )
        i_counts = (
            i_counts
            .select('itemset_i', 'count_i')
            .groupBy('itemset_i')
            .agg(F.first('count_i').alias('count_i'))
            .withColumnRenamed('itemset_i', 'itemset_i_r')
        )
        r_candidates = (
            r_candidates
            .join(i_counts, 
                r_candidates['itemset_i'] == i_counts['itemset_i_r'], 
                how='inner')
            .select('itemset_i', 'item_c', 'itemset_j', 'count', 'count_i')
        )
        r_candidates = (
            r_candidates
            .withColumn('confidence', F.col('count') / F.col('count_i'))
            .where(F.col('confidence') >= min_conf)
        )

        # Update rules with expanded rules
        r_candidates = r_candidates.withColumn('support', F.col('count') / seq_total)
        candidates_format = r_candidates.withColumn('itemset_j', F.concat_ws(', ', 
                                                                r_candidates['itemset_j'], 
                                                                r_candidates['item_c']))
        candidates_format = candidates_format.withColumn('itemset_i', candidates_format['itemset_i'].cast('array<string>'))
        candidates_format = candidates_format.withColumn('itemset_i', F.concat_ws(', ', candidates_format['itemset_i']))
        rules = rules.union(candidates_format.select('itemset_i', 'itemset_j', 'support', 'confidence'))

        if r_candidates.first() == None:
            break

        # Prepare candidates for next loop iteration
        r_candidates = r_candidates.withColumn('item_c', F.array(r_candidates['item_c']))
        r_candidates = (
            r_candidates
            .withColumn('itemset_j', F.array_union(r_candidates['itemset_j'], r_candidates['item_c']))
            .select('itemset_i', 'itemset_j')
        )
    return rules

# COMMAND ----------

## Main algorithm

# Get all base candidate rules to test 
# i.e., all possible rules of one frequent item [i] followed by another frequent item [j]
candidates = (
    freq_items
    .select(F.col('item').alias('item_i'))
    .crossJoin(freq_items.select(F.col('item').alias('item_j')))
)

# Build dataframe that contains, for all candidate rules, 
# all sequences containing the candidate rule 
# where the first occurrence of i is before the last occurrence of j
# (required for a valid rule)
# This dataframe will be used to check candidate rules against the support threshold
candidates = (
    candidates
    .join(item_occurrences, 
        candidates['item_i'] == item_occurrences['item'],
        how='inner')
    .select('item_i', 'item_j', 'seq', 'first')
    .withColumnRenamed('first', 'first_i')
)
right = item_occurrences.withColumnRenamed('seq', 'seq_r')
candidates = (
    candidates
    .join(right, 
        [
            candidates['item_j'] == right['item'], 
            candidates['seq'] == right['seq_r'], 
            candidates['first_i'] < right['last']
        ], 
        how='inner')
    .select('item_i', 'item_j', 'seq')
)

# Check candidates against support threshold
candidates = (
    candidates
    .groupBy(['item_i', 'item_j'])
    .agg(F.count('seq').alias('count'))
    .where(F.col('count') >= min_supp_ct)
)

# Compute confidence for candidates and check against confidence threshold
item_i_counts = (
    item_counts
    .where(item_counts['count'] >= min_supp_ct)
    .withColumnRenamed('count', 'count_i')
)
candidates = (
    candidates
    .join(item_i_counts, 
        candidates['item_i'] == item_i_counts['item'], 
        how='inner')
    .select('item_i', 'item_j', 'count', 'count_i')
)
candidates = (
    candidates
    .withColumn('confidence', F.col('count') / F.col('count_i'))
    .where(F.col('confidence') >= min_conf)
)

# Update rules
candidates = candidates.withColumn('support', F.col('count') / seq_total)
rules = rules.union(candidates.select('item_i', 'item_j', 'support', 'confidence'))

# Convert item_i and item_j to itemset arrays to faciliate recursive looping
candidates = (
    candidates
    .withColumn('itemset_i', F.array(candidates['item_i']))
    .withColumn('itemset_j', F.array(candidates['item_j']))
    .select('itemset_i', 'itemset_j')
)

## Loop to expand the left side of the rules by 1 item at a time
for i in range(1, antecedent_max+1):

    # Inner loop to try to expand the right side of the rules by 1 at a time
    # with the left side of the rules at the current antecedent count
    rules = expand_right(candidates, rules)

    # We have to start running the outer for loop 1 extra time
    # so we can try to expand (on the right) 
    # any rules with antecedent_max items on the left side
    # (hence the antecedent_max+1 for the outer loop iterator range)
    # but then we break before the left side can build beyond antecedent_max
    if i == antecedent_max:
        break

    # Try all frequent items for each rule that are greater than all items in itemset_i
    # This condition is needed to avoid rule duplication in other expansions
    # Add this candidate item (item c) as a new column 
    candidates = candidates.crossJoin(freq_items.withColumnRenamed('item', 'item_c'))
    candidates = candidates.where(F.array_max(candidates['itemset_i']) < candidates['item_c'])

    # Build dataframe of all sequences that contain itemset_i and item_c:
    # "candidates_ic" (for computing confidence later)
    # and dataframe of all sequences that contain the full candidate rule: 
    # "candidates" (for computing support)
    # first_max stores the latest of the first timepoints for each item in 
    # itemset_i plus item_c
    candidates_ic = (
        candidates
        .join(item_occurrences, 
            candidates['itemset_i'][0] == item_occurrences['item'],
            how='inner')
        .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first')
        .withColumnRenamed('first', 'first_max')
    )
    for j in range(1,i):
        candidates_ic = (
            candidates_ic
            .join(right, 
                [
                    candidates_ic['itemset_i'][j] == right['item'],
                    candidates_ic['seq'] == right['seq_r']
                ],
                how='inner')
            .withColumn('first_max', F.greatest('first_max', 'first'))
            .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first_max')
        )
    candidates_ic = (
        candidates_ic
        .join(right, 
            [
                candidates_ic['item_c'] == right['item'], 
                candidates_ic['seq'] == right['seq_r']
            ], 
            how='inner')
        .withColumn('first_max', F.greatest('first_max', 'first'))
        .select('itemset_i', 'item_c', 'itemset_j', 'seq', 'first_max')
    )

    # first_max < last ensures that the item on the right occurs after all items on the left
    candidates = (
        candidates_ic
        .join(right, 
            [
                candidates_ic['itemset_j'][0] == right['item'], 
                candidates_ic['seq'] == right['seq_r'], 
                candidates_ic['first_max'] < right['last'],
            ], 
            how='inner')
        .select('itemset_i', 'item_c', 'itemset_j', 'seq')
    )
    
    # Check candidates against support threshold
    candidates = (
        candidates
        .groupBy(['itemset_i', 'item_c', 'itemset_j'])
        .agg(F.count('seq').alias('count'))
        .where(F.col('count') >= min_supp_ct)
    )

    # Compute confidence for candidates and check against confidence threshold
    ic_counts = (
        candidates_ic
        .select('itemset_i', 'item_c', 'itemset_j', 'seq')
        .groupBy('itemset_i', 'item_c', 'itemset_j')
        .agg(F.count('seq').alias('count_ic'))
    )
    ic_counts = (
        ic_counts
        .select('itemset_i', 'item_c', 'count_ic')
        .groupBy('itemset_i', 'item_c')
        .agg(F.first('count_ic').alias('count_ic'))
        .withColumnRenamed('itemset_i', 'itemset_i_r')
        .withColumnRenamed('item_c', 'item_c_r')
    )
    candidates = (
        candidates
        .join(ic_counts, 
            [
                candidates['itemset_i'] == ic_counts['itemset_i_r'], 
                candidates['item_c'] == ic_counts['item_c_r']
            ], 
            how='inner')
        .select('itemset_i', 'item_c', 'itemset_j', 'count', 'count_ic')
    )
    candidates = (
        candidates
        .withColumn('confidence', F.col('count') / F.col('count_ic'))
        .where(F.col('confidence') >= min_conf)
    )

    # update rules with expanded rules
    candidates = candidates.withColumn('support', F.col('count') / seq_total)
    candidates_format = candidates.withColumn('itemset_i', F.concat_ws(', ', 
                                                            candidates['itemset_i'], 
                                                            candidates['item_c']))
    candidates_format = candidates_format.withColumn('itemset_j', candidates_format['itemset_j'][0].cast('string'))
    rules = rules.union(candidates_format.select('itemset_i', 'itemset_j', 'support', 'confidence'))

    if candidates.first() == None:
        break

    # prepare candidates for next loop iteration
    candidates = candidates.withColumn('item_c', F.array(candidates['item_c']))
    candidates = (
        candidates
        .withColumn('itemset_i', F.array_union(candidates['itemset_i'], candidates['item_c']))
        .select('itemset_i', 'itemset_j')
    )