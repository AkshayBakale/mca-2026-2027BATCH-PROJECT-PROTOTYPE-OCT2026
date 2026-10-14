# PageRank Algorithm - Basic Implementation

# Web pages and their links
pages = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"]
}

# Initial PageRank value
rank = {}
for page in pages:
    rank[page] = 1.0

# Damping factor
d = 0.85

# Number of iterations
iterations = 10

# PageRank calculation
for i in range(iterations):
    new_rank = {}

    for page in pages:
        score = (1 - d)

        # Find pages linking to the current page
        for other_page in pages:
            if page in pages[other_page]:
                # Number of outgoing links
                links = len(pages[other_page])

                score += d * (rank[other_page] / links)

        new_rank[page] = score

    rank = new_rank

# Display PageRank
print("PageRank of Web Pages:")

for page, value in rank.items():
    print(page, ":", round(value, 4))