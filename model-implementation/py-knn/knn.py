from math import sqrt
import numpy as np
from collections import Counter 

def knn(database, query, k=3):
    """ Given a query with n dimensions and a database with n+1 dimensions
    where the n+1 element is the tag. Use the knn algorithm to find the 
    closest point. """
    
    # Calculate the distance
    dist = np.array([sqrt(sum(row)) for row in [(database[:, :-1] - query) ** 2]])

    # Sort the data base by the distance from the point
    sorted_database = database[dist.argsort()]
    
    # Count how many times 
    cnt = Counter()

    for row in sorted_database[:k]:
        cnt[row[-1]] += 1
    
    # Return the key with the closests point to the query 
    return cnt.most_common(1)[0][0]

