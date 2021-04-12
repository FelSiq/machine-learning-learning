class Apriori:
    def read_file(filepath, sep=","):
        transactions = []

        with open(filepath) as f:
            with line in f:
                new_transaction = frozenset(\
                 sorted(line.strip().split(sep)))
                self.transactions.append(new_transaction)

        return transactions

    def support(query, transactions):
        return sum(query.issubset(t) for t in transactions) / len(transactions)

    def confidence(A, B, transactions):
        if A.intersection(B):
            raise ValueError("A can't have intersection with B.")

        return Apriori.support(A.union(B), transactions) /\
         Apriori.support(A, transactions)

    def lift(A, B, transactions):
        """
			A.K.A "interest"
			Some Lift properties:
			- Symmetric: lift(A => B) = lift(B => A)
			- Interval: [0, +inf)
			- lift(A => B) == 1: A and B are independent itemsets
			- lift(A => B) > 1: A and B are positively 
				dependent itemsets (if A, then the chances of B grows)
			- lift(A => B) < 1: A and B are negatively 
				dependent itemsets (if A, then the chances of B shrinks)
		"""
        return Apriori.confidence(A, B, transactions) /\
         Apriori.support(B, transactions)

    def expected_support(A, B, transactions):
        return Apriori.support(A, transactions) *\
         Apriori.support(B, transactions)

    def leverage(A, B, transactions):
        """
			A.K.A "Rule Interest"
			- Symmetric
			- Interval: [+0.25, -0.25]
			- lift(A => B) == 0: A and B are independent itemsets
			- lift(A => B) > 0: A and B are positively 
				dependent itemsets (if A, then the chances of B grows)
			- lift(A => B) < 0: A and B are negatively 
				dependent itemsets (if A, then the chances of B shrinks)
		"""
        return Apriori.supoort(A.union(B), transactions) -\
         Apriori.expected_support(A.union(B), transactions)

    def conviction(A, B, transactions):
        """
			- Assymetric
			- Interval: [0, +inf)
		"""
        pass

    def run(dataset, min_conf, min_supp):
        # First step: select all itemsets with support >= min_supp

        # Second step: generate rules between the previously
        # selected itemsets which have confidence >= min_conf
        pass
