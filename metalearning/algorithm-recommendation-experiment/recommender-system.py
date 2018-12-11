# coding: -*- utf8 -*-
"""
"""
from sklearn.neighbors import KNeighborsClassifier

class MlRecommender:
    def __init__(self):
        """."""
        pass 

    def _average_ranking(self, rankings):
        """."""
        pass

    def load_metadata(self, filepath):
        """."""
        pass

    def _get_baseline(self, rankings):
        """."""
        pass

    def plot(self, performance, baseline):
        """."""
        pass

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 1:
        print("usage:", sys.argv[0], "...")
        exit(1)

    
