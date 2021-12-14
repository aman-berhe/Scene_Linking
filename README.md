# Scene Linking
In this work, scnes are clustered unto their respctive narratives.
Scene might have more than one narrative inside them. Hence, the proposed clustering techinique should consider that.
We have proposed Fuzzy online clustering and graph based community detection of scenes.
Fuzzy online clustering have the ability to cluster scenes into multiple clusters when ever necessary according to a threshold for clustering.
	1. Narrative charcterstics (Characters, Entities, themes, etc.) are used to cluster scenes according to the narrative they convey
	2. Scene trancripts and summaries are also used as textual features (TF-IDF and document embeding, Doc2Vec, are used)
	3. It also incorporates episode level and season level granularities.
	
Graph based commnity detection
	1. Takes the same features as fuzzy online clusters
	2. Scenes are considered as nodes of the graph
	3. Louvin and Dendogram algorithms are used to detct communities 
	
	
	Full Version of the work coming soon..
