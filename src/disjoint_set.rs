/// This is a simple implementation of a Disjoint Set, which allows for
/// efficient operations that involves set union and ancestor finding.
/// We need this to implement the Tarjan algorithm for computing
/// least common ancestors, which is needed to compute the longest common
/// substring.
pub struct DisjointSet {
    ancestors: Vec<usize>,
}

impl DisjointSet {
    pub fn new(size: usize) -> DisjointSet {
        let mut ancestors = Vec::with_capacity(size);
        for i in 0..size {
            // MakeSet(i)
            ancestors.push(i);
        }
        DisjointSet { ancestors }
    }

    pub fn find_set(&mut self, index: usize) -> usize {
        let mut ret = self.ancestors[index];
        if ret != index {
            ret = self.find_set(ret);
            self.ancestors[index] = ret;
        }
        ret
    }

    /// Merge two sets. Always merge `v` into `u` by assuming that `u` has higher rank.
    /// This is not optimum but suitable for  the purpose of this code.
    pub fn union(&mut self, u: usize, v: usize) {
        if self.ancestors[u] == self.ancestors[v] {
            return;
        }
        self.ancestors[v] = u;
    }
}
