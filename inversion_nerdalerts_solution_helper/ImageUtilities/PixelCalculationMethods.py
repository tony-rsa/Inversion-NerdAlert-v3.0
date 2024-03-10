Pairwise distances and Orientation
between = 1
orientations = ["bottom", "top", "left", "right"]
top_ = lambda X,between: X[0: between, :].reshape(1, -1)
right_ = lambda X, between: X[:, 0:between].reshape(1, -1)
bottom_ = lambda X, between: X[(-1-between):-1, :].reshape(1, -1)
left_ = lambda X,between: X[:, (-1-between):-1].reshape(1, -1)

def border_gap(A,B):
    between = 1
    dist1 = pairwise_distances(top_(A,between), bottom_(B,between), metric = "cosine")[0][0]
    dist2 = pairwise_distances(bottom_(A,between), top_(B,between), metric = "cosine")[0][0]
    dist3 = pairwise_distances(right_(A,between), left_(B,between), metric = "cosine")[0][0]
    dist4 = pairwise_distances(left_(A,between), right_(B,between), metric = "cosine")[0][0]
    return [dist1,dist2,dist3,dist4]

def block_distance(A,B):
    ed = border_gap(A,B)
    return np.min(ed), np.argmin(ed)
Individuals Function
def individual(length):
    ui = np.arange(length)
    np.random.shuffle(ui)
    return ui.tolist()
Individual tiles edges in a list
def individual_boundrylist(individual):
    n = int(np.sqrt(len(individual)))
    A = np.array(individual).reshape(n,n)
    observe_edges = set()
    for i in np.arange(n):
        for j in np.arange(n):
            if j+1 < n:
                ii = A[i,j]
                jj = A[i, j+1]
                observe_edges.add((ii,jj,meO[ii, jj]))
            if i+1 < n:
                ii = A[i, j]
                jj = A[i+1, j]
                observe_edges.add((ii, jj, meO[ii, jj]))
    return observe_edges
Fitness score Function
def fitness_score(individual, target = 1):
    elist = individual_boundrylist(individual)
    return (np.sum([me[e[0], e[1]] for e in elist]))
Generate Population and Average Fitness of the Population
def population(count, length):
    return [ individual(length) for x in range(count)]

def category(pop, target):
    summed = reduce(add, (fitness_score(x, target) for x in pop))
    return summed / (len(pop) * 1.0)
Tiles edges in a list with Orientation Information
def boundry_pairs(individual):
    n = int(np.sqrt(len(individual)))
    m = n
    A = np.array(individual).reshape(n,n)
    observe_edges = set()
    for i in np.arange(n):
        for j in np.arange(m):
            if j+1 <m:
                ii = A[i, j]
                jj = A[i, j+1]
                if i < j:
                    obs = (ii, jj, 3.0, ((i,j), (i, j+1)))
                else:
                    obs = (ii, jj, 2.0, ((i,j), (i,j+1)))
                observe_edges.add(obs)
            if i+1 < n:
                ii = A[i, j]
                jj = A[i+1, j]
                if i < j:
                    obs = (ii, jj, 0.0, ((i,j), (i+1, j)))
                else:
                    obs = (ii, jj, 1.0, ((i,j), (i+1, j)))
                    
                observe_edges.add(obs)
    return list(observe_edges)
Crossover of Parents
def crossover(male, female):
    
    child = np.zeros(male.shape[0]).astype(int)
    child[:] = -1
    
    match_indices = np.where(male == female)
    child[match_indices] = male[match_indices]
    
    if child.sum() == -1*child.shape[0]:
        idx = np.random.choice(np.arange(child.shape[0]))
        child[idx] = female[idx]
        
    while (child.sum() < np.arange(child.shape[0]).sum()):
        child = update_boundary(child)
        
    return child
Tiles boundary update after crossover of parents
def update_boundary(child):
    
    n = int(np.sqrt(child.shape[0]))
    Cmat = child.reshape(n,n)
    candidate_pairs = boundry_pairs(child)
    candidate_pairs = [c for c in candidate_pairs if (c[0] == -1) != (c[1] == -1)]
    
    results = []
    for cix in candidate_pairs:
        a = cix[0]
        b = cix[1]
        o = cix[2]
        ref = np.max([a,b])
        n_neighbors = np.argsort(me[ref, :])
        for nn in n_neighbors:
            if ((nn in child) == False) & (meO[ref, nn] == o):
                results.append((ref, nn, o, me[ref, nn], cix[3], (a,b)))
                break
    
    if len(results) > 0:
        result = sample(results, 1)[0]
        if result[5][0] == -1:
            rep_value = result[1]
            ui = int(result[4][0][0])
            yi = int(result[4][0][1])
        else:
            rep_value = result[1]
            ui = int(result[4][1][0])
            yi = int(result[4][1][1])
        
        Cmat[ui, yi] = rep_value
        child = Cmat.reshape(1, -1)[0]
    else:
        idx = np.arange(child.shape[0])
        missing_pieces = idx[np.isin(idx, child) == False]
        np.random.shuffle(missing_pieces)
        missing_pieces
        iter = 0
        for i in np.arange(child.shape[0]):
            if child[i] == -1:
                child[i] = missing_pieces[iter]
                iter += 1
                
    return child
Evolution Function
def evolution(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    
    categoryd = [ (fitness_score(x, target), x) for x in pop]
    categoryd = [ x[1] for x in sorted(categoryd)]
    retain_length = int(len(categoryd)*retain)
    parents = categoryd[:retain_length]
    
    for individual in categoryd[retain_length:]:
        if random_select > random():
            parents.append(individual)
            
    for individual in parents:
        if mutate > random():
            individual = np.roll(individual, 1)
            
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            child = crossover(np.array(male), np.array(female))
            child = child.tolist()
            children.append(child)
            
    parents.extend(children)
    return parents
