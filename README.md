## rule_growth_pyspark  

This repository contains code developed by the TraCS Data Science Lab, which is part of the School of Medicine at the University of North Carolina at Chapel Hill. This code may have been modified from its original form to protect proprietary information and improve interpretability out of context. For example, most paths have been removed.  

## Code Source Environment Notes  
This code was originally developed in a custom secure data enclave platform. It was then reformatted in Azure Databricks to prepare for sharing. Thus, this file retains indicators of breaks between code cells (indicated by # COMMAND ----------).

## Description  
This is a PySpark implementation of the RuleGrowth algorithm for sequential rule mining (originally developed in Java, https://www.philippe-fournier-viger.com/spmf/RuleGrowth.php). This implementation of the algorithm identifies rules for sequences of one item or itemset being followed by another item or itemset within a dataset of sequences. For example, this implementation was developed to explore rules in sequences of diagnoses in patient data. It takes user-defined thresholds for minimum support, minimum confidence, maximum number of antecedents, and maximum number of consequents.  

Upon initially implementing the algorithm in standard Python, performance did not scale well with large amounts of data. This PySpark implementation vastly improves scaling performance.  

## Authors  
This code was developed by John P. Powers as a PySpark implementation of the RuleGrowth algorithm for sequential rule mining (https://www.philippe-fournier-viger.com/spmf/RuleGrowth.php). The original algorithm and code were developed by Philippe Fournier-Viger and colleagues (see reference below). The original code was developed in Java and is available in the SPMF open-source data mining library distributed under the terms of the GNU General Public License v3.0 (https://www.philippe-fournier-viger.com/spmf/). See code file header for more information. 

Fournier-Viger, P., Nkambou, R. & Tseng, V. S. (2011). RuleGrowth: Mining Sequential Rules Common to Several Sequences by Pattern-Growth. Proceedings of the 26th Symposium on Applied Computing (ACM SAC 2011). ACM Press, pp. 954-959.

## Citing This Repository  
If you use this software in your work, please cite it using the CITATION.cff file or by clicking "Cite this repository" on the right. 
