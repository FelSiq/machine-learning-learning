class UnionFindRank:
    """With path compression + union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = n * [0]

    def find(self, x):
        if x != self.parent[x]:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, u, v):
        pu = self.find(u)
        pv = self.find(v)

        if pu == pv:
            return

        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv

        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu

        else:
            self.parent[pu] = pv
            self.rank[pv] += 1

    def unique_ids(self):
        return sorted(set(self.parent))
