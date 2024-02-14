import numpy as np
import networkx as nx

"""
this and mark are used to define a sliding window algorithm. 
    in mark(), the image is flattened with ravel() and then the offsets corresponding to each of the neighbors is calculated
"""
def neighbors(shape):
    dim = len(shape)
    #creates a square shaped matrix, whose dimensions are all 3, and then sets the center to 0
    block = np.ones([3]*dim)
    block[tuple([1]*dim)] = 0

    #get relative positions of the neighbor (from the center)
    idx = np.where(block>0)
    idx = np.array(idx, dtype=np.uint8).T
    idx = np.array(idx-[1]*dim)

    #compute linear indices
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx, acc[::-1])

"""
uses the linear indices of the neighboring elements (calculated in neighbors(shape)) to find all foreground pixels surrounding a given pixel
0 = background (target pixel is ignored if background)
1 = edge
2 = node
"""
# my mark
def mark(img, nbs): # mark the array use (0, 1, 2)
    img = img.ravel()
    for p in range(len(img)):
        #ingnore background elements
        if img[p]==0:continue

        #number of adjacent pixels
        s = 0
        for dp in nbs:
            if img[p+dp]!=0:s+=1
        if s==2:img[p]=1
        else:img[p]=2

# trans index to r, c...
def idx2rc(idx, acc):
    rst = np.zeros((len(idx), len(acc)), dtype=np.int16)
    for i in range(len(idx)):
        for j in range(len(acc)):
            rst[i,j] = idx[i]//acc[j]
            idx[i] -= rst[i,j]*acc[j]
    rst -= 1
    return rst
    
# fill a node (may be two or more points)
def fill(img, p, num, nbs, acc, buf):
    img[p] = num
    buf[0] = p
    cur = 0; s = 1; iso = True;
    
    while True:
        p = buf[cur]
        for dp in nbs:
            cp = p+dp
            if img[cp]==2:
                img[cp] = num
                buf[s] = cp
                s+=1
            if img[cp]==1: iso=False
        cur += 1
        if cur==s:break
    return iso, idx2rc(buf[:s], acc)

# trace the edge and use a buffer, then buf.copy, if use [] numba not works
def trace(img, p, nbs, acc, buf):
    c1 = 0; c2 = 0;
    newp = 0
    cur = 1
    while True:
        buf[cur] = p
        img[p] = 0
        cur += 1
        for dp in nbs:
            cp = p + dp
            if img[cp] >= 10:
                if c1==0:
                    c1 = img[cp]
                    buf[0] = cp
                else:
                    c2 = img[cp]
                    buf[cur] = cp
            if img[cp] == 1:
                newp = cp
        p = newp
        if c2!=0:break
    return (c1-10, c2-10, idx2rc(buf[:cur+1], acc))
   
# parse the image then get the nodes and edges
def parse_struc(img, nbs, acc, iso, ring):
    img = img.ravel()
    buf = np.zeros(131072, dtype=np.int64)
    num = 10
    nodes = []
    for p in range(len(img)):
        if img[p] == 2:
            isiso, nds = fill(img, p, num, nbs, acc, buf)

            if isiso and not iso:
                continue

            num += 1
            nodes.append(nds)
    edges = []
    for p in range(len(img)):
        if img[p] <10: continue
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)

    if not ring: 
        return nodes, edges
    
    for p in range(len(img)):
        if img[p]!=1: continue
        img[p] = num; num += 1
        nodes.append(idx2rc([p], acc))
        for dp in nbs:
            if img[p+dp]==1:
                edge = trace(img, p+dp, nbs, acc, buf)
                edges.append(edge)
    return nodes, edges
    
# use nodes and edges build a networkx graph
"""
    * if full, add the mean centroid to each edge
    * nodes is a list of lists containing the image pixels corresponding to each
"""
def build_graph(nodes, edges, multi=False, full=True):
    os = np.array([i.mean(axis=0) for i in nodes])

    if full: 
        os = os.round().astype(np.uint16)

    graph = nx.MultiGraph() if multi else nx.Graph()

    #add nodes and node pixels of image ('pts')
    for i in range(len(nodes)):
        graph.add_node(i, pts=nodes[i], o=os[i])

    #add nodes and edge pixels ('pts')
    for s, e, pts in edges:
        if full:
            #set first and last elements of the edge pixels
            pts[[0,-1]] = os[[s,e]]

        l = np.linalg.norm(pts[1:]-pts[:-1], axis=1).sum()
        graph.add_edge(s,e, pts=pts, weight=l)
        
    return graph

"""
marks endpoints and notes with more than two neighbors, 1<=n or n>2
"""
def mark_node(ske):
    buf = np.pad(ske, (1,1), mode='constant').astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    return buf
    
def build_sknw(skeleton, multi=False, iso=True, ring=True, full=True):
    """
    Main function to build the graph from a skeletonized image.
    
    Parameters:
    ske (numpy.ndarray): The skeletonized image.
    multi (bool): Create a MultiGraph if True.
    iso (bool): Consider isolated nodes.
    ring (bool): Consider rings with no branching points. will insert a node in this case
    full (bool): if true, every edge starts from the node's centroid, else touch the "node block"
    
    Returns:
    networkx.Graph: where the nodes represent branching and end points, and edges represent the pixels connecting those points
    """
    buf = np.pad(skeleton, (1,1), mode='constant').astype(np.uint16)
    nbs = neighbors(buf.shape)
    acc = np.cumprod((1,)+buf.shape[::-1][:-1])[::-1]
    mark(buf, nbs)
    nodes, edges = parse_struc(buf, nbs, acc, iso, ring)
    return build_graph(nodes, edges, multi, full)
    
# draw the graph
def draw_graph(img, graph, cn=255, ce=128):
    acc = np.cumprod((1,)+img.shape[::-1][:-1])[::-1]
    img = img.ravel()
    for (s, e) in graph.edges():
        eds = graph[s][e]
        if isinstance(graph, nx.MultiGraph):
            for i in eds:
                pts = eds[i]['pts']
                img[np.dot(pts, acc)] = ce
        else: img[np.dot(eds['pts'], acc)] = ce
    for idx in graph.nodes():
        pts = graph.nodes[idx]['pts']
        img[np.dot(pts, acc)] = cn

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    img = np.array([
        [0,0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,1,0],
        [0,0,0,1,0,0,0,0,0],
        [1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,1,0,0,0,1,0,0,0],
        [1,0,1,0,0,1,1,1,1],
        [0,1,0,0,1,0,0,0,0],
        [0,0,0,1,0,0,0,0,0]])

    node_img = mark_node(img)
    para = [{'iso':False}, {'ring':False}, {'full':False},
            {'iso':True}, {'ring':True}, {'full':True}]
    for i,p,k in zip([1,2,3,4,5,6], [231,232,233,234,235,236], para):
        print(k)
        graph = build_sknw(img, False, **k)
        ax = plt.subplot(p)
        ax.imshow(node_img[1:-1,1:-1], cmap='gray')

        # draw edges by pts
        for (s,e) in graph.edges():
            ps = graph[s][e]['pts']
            ax.plot(ps[:,1], ps[:,0], 'green')
            
        # draw node by o
        nodes = graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        ax.plot(ps[:,1], ps[:,0], 'r.')

        # title and show
        ax.title.set_text(k)
    plt.show()
