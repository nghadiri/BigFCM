# BigFCM
Scalable Fuzzy C-Means clustering on Hadoop.

-	There is a growing need for fast clustering of massive datasets on distributed platforms
-	The execution time of existing algorithms is highly increased by increasing the data size
-	We propose a scalable clustering model based on map-reduce and exploit several mechanisms including caching design to achieve several orders of magnitude reduction in execution time
-	We evaluated our algorithm by clustering of several large datasets including SUSY and HIGGS from the UCI repository, and the KDD99 for intrusion detection
-	It performed much faster than Fuzzy K-Means and the execution time increases almost linearly with increased data size
-	The proposed method has almost equal accuracy as traditional clustering methods

Read about fuzzy clustering [here](https://en.wikipedia.org/wiki/Fuzzy_clustering).

Please cite to:
Ghadiri, Nasser, Meysam Ghaffari, and Mohammad Amin Nikbakht. 
"BigFCM: Fast, precise and scalable FCM on hadoop." 
Future Generation Computer Systems 77 (2017): 29-39.
[Link](https://www.sciencedirect.com/science/article/pii/S0167739X17312359) | 
[PDF](https://arxiv.org/pdf/1605.03047)
